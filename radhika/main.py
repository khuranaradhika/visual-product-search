"""
Multimodal Semantic Retrieval Engine
=====================================
HW 2 — Late Fusion Two-Tower Model on Amazon Berkeley Objects (ABO)

Setup:
    1. bash download_abo.sh        # download dataset from S3
    2. pip install -r requirements.txt
    3. python main.py train --max-items 5000
    4. python main.py index --max-items 5000
    5. python main.py eval  --max-items 5000

Requirements:
    pip install torch torchvision transformers faiss-cpu pillow \
                scikit-learn tqdm streamlit
"""

import os, json, glob, gzip, csv, random, math, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0.  CONFIG
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ---- paths (match download_abo.sh layout) ----
    abo_root: str = "data/abo"
    # derived paths (set in __post_init__)
    abo_listings_dir: str = ""
    abo_images_csv: str = ""
    abo_image_dir: str = ""

    # artifacts
    index_save_path: str = "artifacts/faiss_index.bin"
    model_save_path: str = "artifacts/model.pt"
    catalog_ids_path: str = "artifacts/catalog_ids.json"

    # model
    image_encoder: str = "openai/clip-vit-base-patch32"  # 768-d ViT
    text_encoder: str = "openai/clip-vit-base-patch32"   # 512-d text
    image_embed_dim: int = 768    # CLIP ViT hidden size
    text_embed_dim: int = 512     # CLIP text pooler size
    shared_dim: int = 512
    fusion_hidden: int = 1024
    fusion_out: int = 512
    dropout: float = 0.1

    # training
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 20
    temperature: float = 0.07
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0       # 0 is safest on macOS
    seed: int = 42

    # eval
    k_values: list = field(default_factory=lambda: [1, 5, 10])

    # device
    device: str = ""

    def __post_init__(self):
        self.abo_listings_dir = os.path.join(self.abo_root, "listings", "metadata")
        self.abo_images_csv = os.path.join(self.abo_root, "images", "metadata")
        self.abo_image_dir = os.path.join(self.abo_root, "images", "original")
        if not self.device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"


# ---------------------------------------------------------------------------
# 1.  DATASET  (Member A — Data / Pipeline)
# ---------------------------------------------------------------------------

def load_image_id_to_path(cfg: Config) -> dict:
    """
    Parse the ABO images metadata CSV(s) to build a mapping:
        image_id  →  relative file path (e.g. "ab/abcdef.jpg")

    ABO ships:  data/abo/images/metadata/images.csv.gz
    CSV columns: image_id, height, width, path
    """
    mapping = {}
    csv_dir = cfg.abo_images_csv
    csv_files = (
        glob.glob(os.path.join(csv_dir, "*.csv.gz"))
        + glob.glob(os.path.join(csv_dir, "*.csv"))
    )
    if not csv_files:
        print(f"[WARN] No image CSV found in {csv_dir}")
        return mapping

    for cf in csv_files:
        opener = gzip.open if cf.endswith(".gz") else open
        with opener(cf, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iid = row.get("image_id", "").strip()
                path = row.get("path", "").strip()
                if iid and path:
                    mapping[iid] = path

    print(f"[ImageCSV] Loaded {len(mapping)} image_id → path entries")
    return mapping


class ABODataset(Dataset):
    """
    Loads ABO product listings.  Each sample returns:
        - image  : tensor  (3×224×224)
        - text   : str     (concatenated attribute string)
        - item_id: str     (unique product id — ground-truth label)

    ABO metadata is gzipped JSONL.  Each line is one product with fields:
        item_id, brand, material, color, product_type, item_name,
        main_image_id, other_image_id, ...

    Images are resolved via a separate CSV:  image_id → file path.
    """

    def __init__(self, cfg: Config, transform=None, max_items: int = None):
        self.cfg = cfg
        self.transform = transform or self._default_transform()
        self.samples = []  # list of (image_path, attribute_text, item_id)

        # Step 1: build image_id → path lookup
        self.img_lookup = load_image_id_to_path(cfg)

        # Step 2: parse product listings
        self._load_metadata(max_items)

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _extract_text_value(self, field_data, prefer_lang="en_US"):
        """Extract a human-readable string from ABO's language-tagged fields."""
        if field_data is None:
            return None
        if isinstance(field_data, str):
            return field_data
        if isinstance(field_data, list):
            # list of {"language_tag": "en_US", "value": "Oak"} dicts
            # prefer English, fall back to first available
            en_vals = [
                e["value"] for e in field_data
                if isinstance(e, dict) and e.get("language_tag") == prefer_lang
            ]
            if en_vals:
                return ", ".join(en_vals)
            # fallback
            for e in field_data:
                if isinstance(e, dict) and "value" in e:
                    return e["value"]
                if isinstance(e, str):
                    return e
        if isinstance(field_data, dict):
            return field_data.get("value", str(field_data))
        return str(field_data)

    def _extract_attributes(self, item: dict) -> str:
        """Build a single text string from structured metadata fields."""
        parts = []
        for key in ("brand", "material", "color", "product_type",
                     "item_name", "fabric_type", "finish_type", "style"):
            val = self._extract_text_value(item.get(key))
            if val:
                parts.append(f"{key}: {val}")
        return " | ".join(parts) if parts else "unknown product"

    def _resolve_image_path(self, item: dict) -> Optional[str]:
        """Resolve main_image_id → filesystem path via the CSV lookup."""
        main_id = item.get("main_image_id")
        if main_id and main_id in self.img_lookup:
            rel = self.img_lookup[main_id]
            full = os.path.join(self.cfg.abo_image_dir, rel)
            if os.path.isfile(full):
                return full

        # Fallback: try other_image_id list
        for oid in (item.get("other_image_id") or []):
            if oid in self.img_lookup:
                rel = self.img_lookup[oid]
                full = os.path.join(self.cfg.abo_image_dir, rel)
                if os.path.isfile(full):
                    return full
        return None

    def _load_metadata(self, max_items):
        listing_dir = self.cfg.abo_listings_dir
        json_files = sorted(
            glob.glob(os.path.join(listing_dir, "*.json.gz"))
            + glob.glob(os.path.join(listing_dir, "*.json"))
        )
        if not json_files:
            print(f"[ERROR] No listing files found in {listing_dir}")
            print(f"        Have you run download_abo.sh ?")
            return

        count = 0
        skipped = 0
        for jf in json_files:
            opener = gzip.open if jf.endswith(".gz") else open
            with opener(jf, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    img_path = self._resolve_image_path(item)
                    if img_path is None:
                        skipped += 1
                        continue

                    attr_text = self._extract_attributes(item)
                    self.samples.append((
                        img_path, attr_text, item["item_id"]
                    ))
                    count += 1
                    if max_items and count >= max_items:
                        print(f"[ABODataset] Loaded {count} samples "
                              f"(skipped {skipped} w/o images)")
                        return

        print(f"[ABODataset] Loaded {count} samples "
              f"(skipped {skipped} w/o images)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, item_id = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Return a black image on read error
            print(f"[WARN] Failed to open {img_path}: {e}")
            img = Image.new("RGB", (224, 224))
        img = self.transform(img)
        return img, text, item_id


def collate_fn(batch):
    imgs, texts, ids = zip(*batch)
    return torch.stack(imgs), list(texts), list(ids)


def build_dataloaders(cfg: Config, max_items=None):
    ds = ABODataset(cfg, max_items=max_items)
    n = len(ds)
    if n == 0:
        raise RuntimeError(
            "Dataset is empty! Check that:\n"
            "  1. You ran download_abo.sh\n"
            "  2. Listings exist in data/abo/listings/metadata/\n"
            "  3. Images exist in data/abo/images/original/\n"
            "  4. Image CSV exists in data/abo/images/metadata/"
        )
    n_test = max(1, int(n * cfg.test_split))
    n_val = max(1, int(n * cfg.val_split))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise RuntimeError(f"Not enough samples ({n}) for train/val/test split")

    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    kw = dict(batch_size=cfg.batch_size, collate_fn=collate_fn,
              num_workers=cfg.num_workers, pin_memory=(cfg.device != "cpu"))
    return (
        DataLoader(train_ds, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        ds,
    )


# ---------------------------------------------------------------------------
# 2.  MODEL  (Member B — Model / Architecture)
# ---------------------------------------------------------------------------
class ImageTower(nn.Module):
    """Frozen CLIP ViT → pooled visual features → projected to shared_dim."""

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import CLIPVisionModel
        self.backbone = CLIPVisionModel.from_pretrained(cfg.image_encoder)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(cfg.image_embed_dim, cfg.shared_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.backbone(pixel_values=pixel_values)
        pooled = out.pooler_output                      # (B, 768)
        return F.normalize(self.proj(pooled), dim=-1)


class TextTower(nn.Module):
    """CLIP text encoder → 512-d → projected to shared_dim."""

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.text_encoder)
        self.backbone = CLIPTextModel.from_pretrained(cfg.text_encoder)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(cfg.text_embed_dim, cfg.shared_dim)

    def forward(self, texts, device="cpu"):
        tok = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = self.backbone(**tok)
        pooled = out.pooler_output
        return F.normalize(self.proj(pooled), dim=-1)


class FusionHead(nn.Module):
    """Weighted concatenation → MLP → L2-normalised multimodal embedding."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.mlp = nn.Sequential(
            nn.Linear(cfg.shared_dim * 2, cfg.fusion_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, cfg.fusion_out),
        )

    def forward(self, img_emb, txt_emb):
        a = torch.sigmoid(self.alpha)
        weighted = torch.cat([a * img_emb, (1 - a) * txt_emb], dim=-1)
        return F.normalize(self.mlp(weighted), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.image_tower = ImageTower(cfg)
        self.text_tower = TextTower(cfg)
        self.fusion = FusionHead(cfg)

    def forward(self, images, texts):
        img_emb = self.image_tower(images)
        txt_emb = self.text_tower(texts, device=images.device)
        return self.fusion(img_emb, txt_emb)

    def encode_image_only(self, images):
        return self.image_tower(images)

    def encode_text_only(self, texts, device="cpu"):
        return self.text_tower(texts, device)

    def fuse(self, img_emb, txt_emb):
        return self.fusion(img_emb, txt_emb)


# ---------------------------------------------------------------------------
# 3.  LOSS
# ---------------------------------------------------------------------------
class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE / NT-Xent contrastive loss."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, embeddings):
        sim = embeddings @ embeddings.T / self.t
        B = sim.size(0)
        labels = torch.arange(B, device=sim.device)
        return F.cross_entropy(sim, labels)


class PairwiseInfoNCE(nn.Module):
    """Cross-modal contrastive loss between two embedding sets."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, emb_a, emb_b):
        logits = emb_a @ emb_b.T / self.t
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)
        return (F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)) / 2


# ---------------------------------------------------------------------------
# 4.  TRAINING LOOP
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, cfg):
    model.train()
    total_loss = 0
    for imgs, texts, _ in tqdm(loader, desc="Train"):
        imgs = imgs.to(cfg.device)
        embs = model(imgs, texts)
        loss = loss_fn(embs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, loss_fn, cfg):
    model.eval()
    total_loss = 0
    all_embs = []
    for imgs, texts, _ in loader:
        imgs = imgs.to(cfg.device)
        embs = model(imgs, texts)
        total_loss += loss_fn(embs).item()
        all_embs.append(embs.cpu())
    avg_loss = total_loss / max(len(loader), 1)
    all_embs_t = torch.cat(all_embs)
    mean_norm = all_embs_t.norm(dim=-1).mean().item()
    return avg_loss, mean_norm


def train(cfg: Config, max_items=None):
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, full_ds = build_dataloaders(
        cfg, max_items=max_items
    )
    print(f"[Train] {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
    print(f"[Train] Device: {cfg.device}")

    model = TwoTowerModel(cfg).to(cfg.device)
    loss_fn = InfoNCELoss(cfg.temperature)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Train] Trainable parameters: {sum(p.numel() for p in params):,}")
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, cfg)
        v_loss, v_norm = validate(model, val_loader, loss_fn, cfg)
        scheduler.step()
        print(f"Epoch {epoch:3d} | train_loss {t_loss:.4f} | "
              f"val_loss {v_loss:.4f} | emb_norm {v_norm:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"  → saved best model (val_loss={v_loss:.4f})")

    model.load_state_dict(torch.load(cfg.model_save_path, weights_only=True))
    return model, test_loader, full_ds


# ---------------------------------------------------------------------------
# 5.  FAISS INDEX  (Member A cont.)
# ---------------------------------------------------------------------------
def build_faiss_index(model, dataset, cfg: Config):
    import faiss

    model.eval()
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
    )

    all_embs, all_ids = [], []
    with torch.no_grad():
        for imgs, texts, ids in tqdm(loader, desc="Indexing catalog"):
            imgs = imgs.to(cfg.device)
            embs = model(imgs, texts).cpu().numpy()
            all_embs.append(embs)
            all_ids.extend(ids)

    catalog_embs = np.vstack(all_embs).astype("float32")
    faiss.normalize_L2(catalog_embs)
    dim = catalog_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(catalog_embs)

    os.makedirs(os.path.dirname(cfg.index_save_path), exist_ok=True)
    faiss.write_index(index, cfg.index_save_path)
    with open(cfg.catalog_ids_path, "w") as f:
        json.dump(all_ids, f)

    print(f"[FAISS] Built index with {index.ntotal} vectors, dim={dim}")
    return index, all_ids


def load_faiss_index(cfg: Config):
    import faiss
    index = faiss.read_index(cfg.index_save_path)
    with open(cfg.catalog_ids_path) as f:
        ids = json.load(f)
    return index, ids


def query_index(query_emb, index, catalog_ids, k=10):
    import faiss
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)
    results = []
    for q in range(query_emb.shape[0]):
        results.append([
            (float(scores[q, i]), catalog_ids[indices[q, i]])
            for i in range(k)
        ])
    return results


# ---------------------------------------------------------------------------
# 6.  EVALUATION  (Member C)
# ---------------------------------------------------------------------------
def precision_at_k(retrieved_ids, gt_id, k):
    return sum(1 for r in retrieved_ids[:k] if r == gt_id) / k

def recall_at_k(retrieved_ids, gt_id, k):
    return 1.0 if gt_id in retrieved_ids[:k] else 0.0

def average_precision(retrieved_ids, gt_id):
    for rank, rid in enumerate(retrieved_ids, 1):
        if rid == gt_id:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(model, test_loader, index, catalog_ids, cfg):
    model.eval()
    metrics = {f"P@{k}": [] for k in cfg.k_values}
    metrics.update({f"R@{k}": [] for k in cfg.k_values})
    all_aps = []
    max_k = max(cfg.k_values)

    with torch.no_grad():
        for imgs, texts, gt_ids in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(cfg.device)
            embs = model(imgs, texts).cpu().numpy().astype("float32")
            batch_results = query_index(embs, index, catalog_ids, k=max_k)

            for i, gt_id in enumerate(gt_ids):
                retrieved = [r[1] for r in batch_results[i]]
                for k in cfg.k_values:
                    metrics[f"P@{k}"].append(precision_at_k(retrieved, gt_id, k))
                    metrics[f"R@{k}"].append(recall_at_k(retrieved, gt_id, k))
                all_aps.append(average_precision(retrieved, gt_id))

    results = {name: np.mean(vals) for name, vals in metrics.items()}
    results["MAP"] = np.mean(all_aps)

    print("\n===== Retrieval Metrics =====")
    for name, val in results.items():
        print(f"  {name:8s}: {val:.4f}")
    return results


# ---------------------------------------------------------------------------
# 7.  MODALITY GAP
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_modality_gap(model, loader, cfg, n_batches=20):
    model.eval()
    cosines = []
    for i, (imgs, texts, _) in enumerate(loader):
        if i >= n_batches:
            break
        imgs = imgs.to(cfg.device)
        img_emb = model.encode_image_only(imgs)
        txt_emb = model.encode_text_only(texts, cfg.device)
        cos = F.cosine_similarity(img_emb, txt_emb, dim=-1)
        cosines.append(cos.cpu())
    cosines = torch.cat(cosines)
    gap = 1.0 - cosines.mean().item()
    print(f"[Modality Gap] mean cosine = {cosines.mean():.4f}, gap = {gap:.4f}")
    return gap


# ---------------------------------------------------------------------------
# 8.  STREAMLIT APP CODE  (save as app.py)
# ---------------------------------------------------------------------------
STREAMLIT_APP_CODE = r'''
"""
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import torch, json, faiss
from PIL import Image
from torchvision import transforms
from main import Config, TwoTowerModel, query_index

st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("Multimodal Semantic Product Search")

@st.cache_resource
def load_resources():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(torch.load(cfg.model_save_path,
                          map_location=cfg.device, weights_only=True))
    model.eval()
    index = faiss.read_index(cfg.index_save_path)
    with open(cfg.catalog_ids_path) as f:
        catalog_ids = json.load(f)
    return cfg, model, index, catalog_ids

cfg, model, index, catalog_ids = load_resources()

with st.sidebar:
    st.header("Query Inputs")
    uploaded = st.file_uploader("Product Image", type=["jpg","jpeg","png"])
    brand    = st.text_input("Brand", placeholder="e.g. Rivet")
    material = st.text_input("Material", placeholder="e.g. Oak")
    color    = st.text_input("Color", placeholder="e.g. Natural")
    ptype    = st.text_input("Product Type", placeholder="e.g. Table")
    top_k    = st.slider("Top-K results", 1, 20, 5)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query image", width=250)
    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])
    img_t = tfm(img).unsqueeze(0).to(cfg.device)
    parts = []
    if brand:    parts.append(f"brand: {brand}")
    if material: parts.append(f"material: {material}")
    if color:    parts.append(f"color: {color}")
    if ptype:    parts.append(f"product_type: {ptype}")
    query_text = " | ".join(parts) if parts else "unknown product"
    st.markdown(f"**Attribute query:** `{query_text}`")
    if st.button("Search"):
        with torch.no_grad():
            emb = model(img_t, [query_text]).cpu().numpy().astype("float32")
        results = query_index(emb, index, catalog_ids, k=top_k)[0]
        st.subheader(f"Top-{top_k} Results")
        for i, (score, pid) in enumerate(results):
            st.write(f"**#{i+1}** — score: {score:.3f} — item_id: `{pid}`")
'''


# ---------------------------------------------------------------------------
# 9.  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Semantic Retrieval Engine"
    )
    parser.add_argument(
        "stage",
        choices=["train", "index", "eval", "gap", "demo_query", "check_data"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--max-items", type=int, default=None,
                        help="Cap dataset size (for quick debugging)")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to query image (demo_query)")
    parser.add_argument("--text", type=str, default="",
                        help="Attribute text for query (demo_query)")
    args = parser.parse_args()
    cfg = Config()

    # ------------------------------------------------------------------
    if args.stage == "check_data":
        """Diagnostic: verify data paths exist and show sample."""
        print(f"ABO root:       {cfg.abo_root}")
        print(f"Listings dir:   {cfg.abo_listings_dir}")
        print(f"  exists?       {os.path.isdir(cfg.abo_listings_dir)}")
        lfiles = glob.glob(os.path.join(cfg.abo_listings_dir, "*"))
        print(f"  files:        {lfiles[:5]}")
        print(f"Images CSV dir: {cfg.abo_images_csv}")
        print(f"  exists?       {os.path.isdir(cfg.abo_images_csv)}")
        cfiles = glob.glob(os.path.join(cfg.abo_images_csv, "*"))
        print(f"  files:        {cfiles[:5]}")
        print(f"Images dir:     {cfg.abo_image_dir}")
        print(f"  exists?       {os.path.isdir(cfg.abo_image_dir)}")
        ifiles = glob.glob(os.path.join(cfg.abo_image_dir, "**/*.*"),
                           recursive=True)
        print(f"  sample imgs:  {ifiles[:3]}")
        print(f"  total imgs:   {len(ifiles)}")

        # Try loading a few samples
        if lfiles and cfiles:
            print("\nAttempting to load 5 samples...")
            ds = ABODataset(cfg, max_items=5)
            for i, (img, txt, iid) in enumerate(ds):
                print(f"  [{i}] id={iid}  shape={img.shape}  text={txt[:80]}...")
        return

    # ------------------------------------------------------------------
    if args.stage == "train":
        model, test_loader, full_ds = train(cfg, max_items=args.max_items)
        print("Training complete. Run 'index' next.")

    elif args.stage == "index":
        model = TwoTowerModel(cfg).to(cfg.device)
        model.load_state_dict(
            torch.load(cfg.model_save_path, weights_only=True,
                        map_location=cfg.device)
        )
        full_ds = ABODataset(cfg, max_items=args.max_items)
        build_faiss_index(model, full_ds, cfg)

    elif args.stage == "eval":
        model = TwoTowerModel(cfg).to(cfg.device)
        model.load_state_dict(
            torch.load(cfg.model_save_path, weights_only=True,
                        map_location=cfg.device)
        )
        _, _, test_loader, _ = build_dataloaders(cfg, max_items=args.max_items)
        index, catalog_ids = load_faiss_index(cfg)
        evaluate_retrieval(model, test_loader, index, catalog_ids, cfg)

    elif args.stage == "gap":
        model = TwoTowerModel(cfg).to(cfg.device)
        model.load_state_dict(
            torch.load(cfg.model_save_path, weights_only=True,
                        map_location=cfg.device)
        )
        _, val_loader, _, _ = build_dataloaders(cfg, max_items=args.max_items)
        compute_modality_gap(model, val_loader, cfg)

    elif args.stage == "demo_query":
        assert args.image, "Provide --image path"
        model = TwoTowerModel(cfg).to(cfg.device)
        model.load_state_dict(
            torch.load(cfg.model_save_path, weights_only=True,
                        map_location=cfg.device)
        )
        model.eval()
        index, catalog_ids = load_faiss_index(cfg)
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
        ])
        img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0)
        img = img.to(cfg.device)
        text = args.text or "unknown product"
        with torch.no_grad():
            emb = model(img, [text]).cpu().numpy().astype("float32")
        results = query_index(emb, index, catalog_ids, k=10)[0]
        print("\nTop-10 results:")
        for rank, (score, pid) in enumerate(results, 1):
            print(f"  {rank:2d}. score={score:.4f}  item_id={pid}")


if __name__ == "__main__":
    main()