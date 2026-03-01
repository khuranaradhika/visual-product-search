"""
Multimodal Semantic Retrieval Engine
=====================================
HW 2 — Late Fusion Two-Tower Model on Amazon Berkeley Objects (ABO)

Architecture:
    Image Tower : DINOv3 ViT-B/16 (last N blocks unfrozen) → 768-d → 512-d
    Text  Tower : CLIP text (last N blocks unfrozen) + attribute-aware
                  encoding with learned attention pooling → 512-d
    Fusion      : weighted concat → MLP → 512-d multimodal embedding

Changes from baseline:
    1. Partial unfreezing — last 2 transformer blocks of each backbone
       are trainable with a lower learning rate (10x smaller).
    2. Attribute-aware text encoding — each "key: value" attribute is
       encoded independently by CLIP-text, then aggregated via a small
       learned attention pooling layer. This prevents long concatenated
       strings from diluting individual attribute signals.

Setup:
    1. bash download_abo.sh
    2. pip install -r requirements.txt
    3. python main.py train --max-items 5000
    4. python main.py index --max-items 5000
    5. python main.py eval  --max-items 5000
"""

import os, json, glob, gzip, csv, random, math, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0.  CONFIG
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ---- paths ----
    abo_root: str = "data/abo"
    abo_listings_dir: str = ""
    abo_images_csv: str = ""
    abo_image_dir: str = ""

    # artifacts
    index_save_path: str = "artifacts/faiss_index.bin"
    model_save_path: str = "artifacts/model.pt"
    catalog_ids_path: str = "artifacts/catalog_ids.json"

    # model
    image_encoder: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    text_encoder: str = "openai/clip-vit-base-patch32"
    image_embed_dim: int = 768
    text_embed_dim: int = 512
    shared_dim: int = 512
    fusion_hidden: int = 1024
    fusion_out: int = 512
    dropout: float = 0.1

    # partial unfreezing
    unfreeze_image_layers: int = 2   # last N DINOv2 transformer blocks
    unfreeze_text_layers: int = 2    # last N CLIP text transformer blocks
    backbone_lr_scale: float = 0.1   # backbone LR = base LR × this

    # attribute-aware encoding
    max_attributes: int = 8          # max separate attributes to encode
    attr_attn_heads: int = 4         # attention heads in attribute pooler

    # training
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 15
    temperature: float = 0.07
    val_split: float = 0.1
    test_split: float = 0.1
    num_workers: int = 0
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
            self.device = "cpu"


# ---------------------------------------------------------------------------
# 1.  DATASET  (Member A — Data / Pipeline)
# ---------------------------------------------------------------------------

def load_image_id_to_path(cfg: Config) -> dict:
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


def parse_attributes(attr_text: str) -> List[str]:
    """
    Split 'brand: X | material: Y | color: Z' into individual
    attribute strings: ['brand: X', 'material: Y', 'color: Z'].
    """
    parts = [p.strip() for p in attr_text.split(" | ") if p.strip()]
    return parts if parts else ["unknown product"]


class ABODataset(Dataset):
    """
    Each sample returns:
        image      : tensor     (3×224×224)
        text       : str        (full concatenated attributes for backward compat)
        attributes : list[str]  (individual attribute strings)
        item_id    : str        (unique product id)
    """

    def __init__(self, cfg: Config, transform=None, max_items: int = None):
        self.cfg = cfg
        self.transform = transform or self._default_transform()
        self.samples = []  # (img_path, attr_text, item_id)
        self.img_lookup = load_image_id_to_path(cfg)
        self._load_metadata(max_items)

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    @staticmethod
    def _train_transform():
        """Augmented transform for training — improves generalization."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _extract_text_value(self, field_data, prefer_lang="en_US"):
        if field_data is None:
            return None
        if isinstance(field_data, str):
            return field_data
        if isinstance(field_data, list):
            en_vals = [
                e["value"] for e in field_data
                if isinstance(e, dict) and e.get("language_tag") == prefer_lang
            ]
            if en_vals:
                return ", ".join(en_vals)
            for e in field_data:
                if isinstance(e, dict) and "value" in e:
                    return e["value"]
                if isinstance(e, str):
                    return e
        if isinstance(field_data, dict):
            return field_data.get("value", str(field_data))
        return str(field_data)

    def _extract_attributes(self, item: dict) -> str:
        parts = []
        for key in ("brand", "material", "color", "product_type",
                     "item_name", "fabric_type", "finish_type", "style"):
            val = self._extract_text_value(item.get(key))
            if val:
                parts.append(f"{key}: {val}")
        return " | ".join(parts) if parts else "unknown product"

    def _resolve_image_path(self, item: dict) -> Optional[str]:
        main_id = item.get("main_image_id")
        if main_id and main_id in self.img_lookup:
            rel = self.img_lookup[main_id]
            full = os.path.join(self.cfg.abo_image_dir, rel)
            if os.path.isfile(full):
                return full
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
            print(f"[ERROR] No listing files in {listing_dir}")
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
                    self.samples.append((img_path, attr_text, item["item_id"]))
                    count += 1
                    if max_items and count >= max_items:
                        print(f"[ABODataset] Loaded {count} (skipped {skipped})")
                        return
        print(f"[ABODataset] Loaded {count} (skipped {skipped})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text, item_id = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {img_path}: {e}")
            img = Image.new("RGB", (224, 224))
        img = self.transform(img)
        attributes = parse_attributes(text)
        return img, text, attributes, item_id


def collate_fn(batch):
    imgs, texts, attr_lists, ids = zip(*batch)
    return torch.stack(imgs), list(texts), list(attr_lists), list(ids)


def build_dataloaders(cfg: Config, max_items=None):
    ds = ABODataset(cfg, max_items=max_items)
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset empty — check download_abo.sh and paths")
    n_test = max(1, int(n * cfg.test_split))
    n_val = max(1, int(n * cfg.val_split))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise RuntimeError(f"Not enough samples ({n}) for splits")
    train_ds, val_ds, test_ds = random_split(
        ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Wrap training subset with augmented transforms
    class AugmentedSubset(Dataset):
        """Re-loads images with augmented transform for training."""
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            orig_idx = self.subset.indices[idx]
            img_path, text, item_id = self.subset.dataset.samples[orig_idx]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            img = self.transform(img)
            attributes = parse_attributes(text)
            return img, text, attributes, item_id

    train_aug = AugmentedSubset(train_ds, ABODataset._train_transform())

    kw = dict(batch_size=cfg.batch_size, collate_fn=collate_fn,
              num_workers=cfg.num_workers, pin_memory=False)
    return (
        DataLoader(train_aug, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        ds,
    )


# ---------------------------------------------------------------------------
# 2.  MODEL  (Member B — Model / Architecture)
# ---------------------------------------------------------------------------

# ---- helpers for partial unfreezing ----

def _freeze_all(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def _unfreeze_last_n_blocks(encoder_layers: nn.ModuleList, n: int):
    """Unfreeze the last `n` transformer blocks in a layer list."""
    total = len(encoder_layers)
    for i, block in enumerate(encoder_layers):
        if i >= total - n:
            for p in block.parameters():
                p.requires_grad = True


class ImageTower(nn.Module):
    """
    DINOv2 ViT-B/14 with last N blocks unfrozen.
    CLS token → projected to shared_dim.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import AutoModel
        print(f"[ImageTower] Loading DINOv3 from {cfg.image_encoder} "
              f"(unfreezing last {cfg.unfreeze_image_layers} blocks)...")
        self.backbone = AutoModel.from_pretrained(cfg.image_encoder)

        # Freeze everything first, then selectively unfreeze
        _freeze_all(self.backbone)
        # DINOv3 uses encoder.layer just like DINOv2
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            _unfreeze_last_n_blocks(
                self.backbone.encoder.layer, cfg.unfreeze_image_layers
            )
        # Unfreeze the final layernorm
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        self.proj = nn.Linear(cfg.image_embed_dim, cfg.shared_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        # DINOv3 provides pooler_output; fall back to CLS token if not
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0]
        return F.normalize(self.proj(pooled), dim=-1)


class AttributeAttentionPooler(nn.Module):
    """
    Given a variable number of per-attribute CLIP embeddings, aggregate
    them with learned multi-head attention into a single vector.

    Uses a learnable [QUERY] token that attends over per-attribute
    embeddings to produce one pooled representation.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, attr_embs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        attr_embs: (B, max_attrs, D)  — per-attribute CLIP embeddings
        mask:      (B, max_attrs)      — True where attribute is padding
        Returns:   (B, D)
        """
        B = attr_embs.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        pooled, _ = self.attn(q, attr_embs, attr_embs, key_padding_mask=mask)
        return self.norm(pooled.squeeze(1))  # (B, D)


class TextTower(nn.Module):
    """
    CLIP text encoder (last N blocks unfrozen) with attribute-aware encoding.

    Instead of encoding one long concatenated string, each attribute
    (e.g. 'brand: Rivet', 'material: Oak') is encoded separately,
    then aggregated via learned attention pooling.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        print(f"[TextTower] Loading CLIP text from {cfg.text_encoder} "
              f"(unfreezing last {cfg.unfreeze_text_layers} blocks)...")
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.text_encoder)
        self.backbone = CLIPTextModel.from_pretrained(cfg.text_encoder)

        # Freeze everything, then unfreeze last N layers
        _freeze_all(self.backbone)
        _unfreeze_last_n_blocks(
            self.backbone.text_model.encoder.layers, cfg.unfreeze_text_layers
        )
        # Unfreeze final layernorm
        if hasattr(self.backbone.text_model, "final_layer_norm"):
            for p in self.backbone.text_model.final_layer_norm.parameters():
                p.requires_grad = True

        self.max_attrs = cfg.max_attributes
        self.pooler = AttributeAttentionPooler(
            cfg.text_embed_dim, cfg.attr_attn_heads, cfg.dropout
        )
        self.proj = nn.Linear(cfg.text_embed_dim, cfg.shared_dim)

    def _encode_strings(self, strings: List[str], device) -> torch.Tensor:
        """Encode a flat list of strings → (len, D) embeddings."""
        tok = self.tokenizer(
            strings, padding=True, truncation=True,
            max_length=77, return_tensors="pt"
        ).to(device)
        out = self.backbone(**tok)
        return out.pooler_output  # (len, 512)

    def forward(self, attr_lists: List[List[str]], device="cpu") -> torch.Tensor:
        """
        attr_lists: list of B items, each is a list of attribute strings.
        Returns: (B, shared_dim)
        """
        B = len(attr_lists)
        D = self.backbone.config.hidden_size  # 512
        M = self.max_attrs

        # Flatten all attributes into one list for a single forward pass
        flat_strs = []
        # Track (batch_idx, attr_idx) for each flat string
        positions = []
        for b, attrs in enumerate(attr_lists):
            for a, attr_str in enumerate(attrs[:M]):
                flat_strs.append(attr_str)
                positions.append((b, a))

        if not flat_strs:
            # Edge case: no attributes at all
            dummy = torch.zeros(B, self.proj.out_features, device=device)
            return F.normalize(dummy, dim=-1)

        # Single batched forward pass through CLIP text
        flat_embs = self._encode_strings(flat_strs, device)  # (N_total, D)

        # Scatter into padded (B, M, D) tensor
        attr_tensor = torch.zeros(B, M, D, device=device)
        pad_mask = torch.ones(B, M, dtype=torch.bool, device=device)  # True=pad
        for idx, (b, a) in enumerate(positions):
            attr_tensor[b, a] = flat_embs[idx]
            pad_mask[b, a] = False

        # Attention-pool across attributes → (B, D)
        pooled = self.pooler(attr_tensor, pad_mask)
        return F.normalize(self.proj(pooled), dim=-1)


class FusionHead(nn.Module):
    """Weighted concat → MLP → L2-normalised multimodal embedding."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5 → balanced start
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

    def forward(self, images, attr_lists):
        img_emb = self.image_tower(images)
        txt_emb = self.text_tower(attr_lists, device=images.device)
        return self.fusion(img_emb, txt_emb)

    def encode_image_only(self, images):
        return self.image_tower(images)

    def encode_text_only(self, attr_lists, device="cpu"):
        return self.text_tower(attr_lists, device)

    def fuse(self, img_emb, txt_emb):
        return self.fusion(img_emb, txt_emb)


# ---------------------------------------------------------------------------
# 3.  LOSS
# ---------------------------------------------------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, embeddings):
        sim = embeddings @ embeddings.T / self.t
        B = sim.size(0)
        labels = torch.arange(B, device=sim.device)
        return F.cross_entropy(sim, labels)


class PairwiseInfoNCE(nn.Module):
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
# 4.  TRAINING LOOP  (with differential learning rates)
# ---------------------------------------------------------------------------

def _build_param_groups(model: TwoTowerModel, cfg: Config):
    """
    Three param groups with differential LR:
      1. Unfrozen backbone layers   → lr × backbone_lr_scale  (0.1×)
      2. Projection layers + pooler → lr                      (1×)
      3. Fusion head                → lr                      (1×)
    """
    backbone_params = []
    head_params = []

    # Image tower: unfrozen backbone params vs projection
    for name, p in model.image_tower.backbone.named_parameters():
        if p.requires_grad:
            backbone_params.append(p)
    head_params.extend(model.image_tower.proj.parameters())

    # Text tower: unfrozen backbone params vs projection + pooler
    for name, p in model.text_tower.backbone.named_parameters():
        if p.requires_grad:
            backbone_params.append(p)
    head_params.extend(model.text_tower.pooler.parameters())
    head_params.extend(model.text_tower.proj.parameters())

    # Fusion head
    head_params.extend(model.fusion.parameters())

    return [
        {"params": backbone_params, "lr": cfg.lr * cfg.backbone_lr_scale},
        {"params": head_params, "lr": cfg.lr},
    ]


def train_one_epoch(model, loader, optimizer, fusion_loss_fn, align_loss_fn, cfg):
    model.train()
    total_loss = 0
    for imgs, texts, attr_lists, _ in tqdm(loader, desc="Train"):
        imgs = imgs.to(cfg.device)
        img_emb = model.image_tower(imgs)
        txt_emb = model.text_tower(attr_lists, device=cfg.device)
        fused = model.fusion(img_emb, txt_emb)

        l_fusion = fusion_loss_fn(fused)
        l_align = align_loss_fn(img_emb, txt_emb)
        loss = l_fusion + 1.0 * l_align

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to protect unfrozen backbone layers
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, fusion_loss_fn, align_loss_fn, cfg):
    model.eval()
    total_loss = 0
    cosines = []
    for imgs, texts, attr_lists, _ in loader:
        imgs = imgs.to(cfg.device)
        img_emb = model.image_tower(imgs)
        txt_emb = model.text_tower(attr_lists, device=cfg.device)
        fused = model.fusion(img_emb, txt_emb)

        l_fusion = fusion_loss_fn(fused)
        l_align = align_loss_fn(img_emb, txt_emb)
        total_loss += (l_fusion + 1.0 * l_align).item()

        cos = F.cosine_similarity(img_emb, txt_emb, dim=-1)
        cosines.append(cos.cpu())

    avg_loss = total_loss / max(len(loader), 1)
    mean_cos = torch.cat(cosines).mean().item()
    return avg_loss, mean_cos


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
    fusion_loss_fn = InfoNCELoss(cfg.temperature)
    align_loss_fn = PairwiseInfoNCE(cfg.temperature)

    # Count parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_trainable = sum(
        p.numel() for n, p in model.image_tower.backbone.named_parameters()
        if p.requires_grad
    ) + sum(
        p.numel() for n, p in model.text_tower.backbone.named_parameters()
        if p.requires_grad
    )
    print(f"[Train] Trainable params: {total_trainable:,} "
          f"(backbone: {backbone_trainable:,}, "
          f"heads: {total_trainable - backbone_trainable:,})")

    param_groups = _build_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t_loss = train_one_epoch(model, train_loader, optimizer,
                                 fusion_loss_fn, align_loss_fn, cfg)
        v_loss, v_cos = validate(model, val_loader,
                                 fusion_loss_fn, align_loss_fn, cfg)
        scheduler.step()
        # Show both learning rates
        lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
        print(f"Epoch {epoch:3d} | train_loss {t_loss:.4f} | "
              f"val_loss {v_loss:.4f} | val_cos {v_cos:.4f} | "
              f"lr={lrs}")
        if v_loss < best_val:
            best_val = v_loss
            os.makedirs(os.path.dirname(cfg.model_save_path), exist_ok=True)
            torch.save(model.state_dict(), cfg.model_save_path)
            print(f"  → saved best model (val_loss={v_loss:.4f})")

    model.load_state_dict(torch.load(cfg.model_save_path, weights_only=True))
    return model, test_loader, full_ds


# ---------------------------------------------------------------------------
# 5.  INDEX (numpy brute-force)
# ---------------------------------------------------------------------------
def build_faiss_index(model, dataset, cfg: Config):
    model.eval()
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
    )
    all_embs, all_ids = [], []
    with torch.no_grad():
        for imgs, texts, attr_lists, ids in tqdm(loader, desc="Indexing catalog"):
            imgs = imgs.to(cfg.device)
            embs = model(imgs, attr_lists).cpu().numpy()
            all_embs.append(embs)
            all_ids.extend(ids)
    catalog_embs = np.vstack(all_embs).astype("float32")
    norms = np.linalg.norm(catalog_embs, axis=1, keepdims=True)
    catalog_embs = catalog_embs / np.maximum(norms, 1e-8)
    os.makedirs(os.path.dirname(cfg.index_save_path), exist_ok=True)
    embs_path = cfg.index_save_path.replace(".bin", "_embs.npy")
    np.save(embs_path, catalog_embs)
    with open(cfg.catalog_ids_path, "w") as f:
        json.dump(all_ids, f)
    print(f"[Index] Saved {catalog_embs.shape[0]} vectors "
          f"(dim={catalog_embs.shape[1]}) to {embs_path}")
    return catalog_embs, all_ids


def load_faiss_index(cfg: Config):
    embs_path = cfg.index_save_path.replace(".bin", "_embs.npy")
    catalog_embs = np.load(embs_path)
    with open(cfg.catalog_ids_path) as f:
        ids = json.load(f)
    return catalog_embs, ids


def query_index(query_emb, catalog_embs, catalog_ids, k=10):
    norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.maximum(norms, 1e-8)
    sims = query_emb @ catalog_embs.T
    results = []
    for q in range(query_emb.shape[0]):
        topk_idx = np.argsort(sims[q])[::-1][:k]
        results.append([(float(sims[q, i]), catalog_ids[i]) for i in topk_idx])
    return results


# ---------------------------------------------------------------------------
# 6.  EVALUATION  (Member C)
# ---------------------------------------------------------------------------
def precision_at_k(retrieved_labels, gt_label, k):
    return sum(1 for r in retrieved_labels[:k] if r == gt_label) / k

def recall_at_k(retrieved_labels, gt_label, k, total_relevant=1):
    found = sum(1 for r in retrieved_labels[:k] if r == gt_label)
    return found / total_relevant

def average_precision(retrieved_labels, gt_label):
    hits = 0
    sum_prec = 0.0
    for rank, rl in enumerate(retrieved_labels, 1):
        if rl == gt_label:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / max(hits, 1) if hits > 0 else 0.0


def _extract_product_type(attr_text: str) -> str:
    for part in attr_text.split(" | "):
        if part.strip().lower().startswith("product_type:"):
            return part.split(":", 1)[1].strip().upper()
    return "UNKNOWN"


def evaluate_retrieval(model, test_loader, catalog_embs, catalog_ids,
                       catalog_attrs, cfg):
    model.eval()
    max_k = max(cfg.k_values)

    catalog_types = {
        iid: _extract_product_type(catalog_attrs.get(iid, ""))
        for iid in catalog_ids
    }

    metrics = {f"P@{k}": [] for k in cfg.k_values}
    metrics.update({f"R@{k}": [] for k in cfg.k_values})
    all_aps = []

    with torch.no_grad():
        for imgs, texts, attr_lists, gt_ids in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(cfg.device)
            embs = model(imgs, attr_lists).cpu().numpy().astype("float32")
            batch_results = query_index(embs, catalog_embs, catalog_ids, k=max_k)

            for i, (gt_id, query_text) in enumerate(zip(gt_ids, texts)):
                gt_type = _extract_product_type(query_text)
                retrieved_ids = [r[1] for r in batch_results[i]]
                retrieved_types = [
                    catalog_types.get(rid, "NONE") for rid in retrieved_ids
                ]
                total_relevant = max(
                    sum(1 for t in catalog_types.values() if t == gt_type), 1
                )
                for k in cfg.k_values:
                    metrics[f"P@{k}"].append(
                        precision_at_k(retrieved_types, gt_type, k)
                    )
                    metrics[f"R@{k}"].append(
                        recall_at_k(retrieved_types, gt_type, k, total_relevant)
                    )
                all_aps.append(average_precision(retrieved_types, gt_type))

    results = {name: np.mean(vals) for name, vals in metrics.items()}
    results["MAP"] = np.mean(all_aps)

    print("\n===== Retrieval Metrics (by product type) =====")
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
    for i, (imgs, texts, attr_lists, _) in enumerate(loader):
        if i >= n_batches:
            break
        imgs = imgs.to(cfg.device)
        img_emb = model.encode_image_only(imgs)
        txt_emb = model.encode_text_only(attr_lists, cfg.device)
        cos = F.cosine_similarity(img_emb, txt_emb, dim=-1)
        cosines.append(cos.cpu())
    cosines = torch.cat(cosines)
    gap = 1.0 - cosines.mean().item()
    print(f"[Modality Gap] mean cosine = {cosines.mean():.4f}, gap = {gap:.4f}")
    return gap


# ---------------------------------------------------------------------------
# 8.  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Semantic Retrieval Engine"
    )
    parser.add_argument(
        "stage",
        choices=["train", "index", "eval", "gap", "demo_query", "check_data"],
    )
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--text", type=str, default="")
    args = parser.parse_args()
    cfg = Config()

    # ------------------------------------------------------------------
    if args.stage == "check_data":
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
        if lfiles and cfiles:
            print("\nLoading 5 samples...")
            ds = ABODataset(cfg, max_items=5)
            for i, (img, txt, attrs, iid) in enumerate(ds):
                print(f"  [{i}] id={iid}  shape={img.shape}")
                print(f"       attrs={attrs}")
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
        train_loader, val_loader, test_loader, full_ds = build_dataloaders(
            cfg, max_items=args.max_items
        )
        catalog_attrs = {
            iid: attr for _, attr, iid in full_ds.samples
        }
        index_ds = ConcatDataset([train_loader.dataset, val_loader.dataset])
        index_loader = DataLoader(
            index_ds, batch_size=cfg.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=cfg.num_workers, pin_memory=False,
        )
        model.eval()
        all_embs, all_ids = [], []
        with torch.no_grad():
            for imgs, texts, attr_lists, ids in tqdm(index_loader,
                                                      desc="Building catalog"):
                imgs = imgs.to(cfg.device)
                embs = model(imgs, attr_lists).cpu().numpy()
                all_embs.append(embs)
                all_ids.extend(ids)
        catalog_embs = np.vstack(all_embs).astype("float32")
        norms = np.linalg.norm(catalog_embs, axis=1, keepdims=True)
        catalog_embs = catalog_embs / np.maximum(norms, 1e-8)
        evaluate_retrieval(model, test_loader, catalog_embs, all_ids,
                           catalog_attrs, cfg)

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
        catalog_embs, catalog_ids = load_faiss_index(cfg)
        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
        ])
        img = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0)
        img = img.to(cfg.device)
        text = args.text or "unknown product"
        query_attrs = parse_attributes(text)
        with torch.no_grad():
            emb = model(img, [query_attrs]).cpu().numpy().astype("float32")
        results = query_index(emb, catalog_embs, catalog_ids, k=10)[0]
        print("\nTop-10 results:")
        for rank, (score, pid) in enumerate(results, 1):
            print(f"  {rank:2d}. score={score:.4f}  item_id={pid}")


if __name__ == "__main__":
    main()