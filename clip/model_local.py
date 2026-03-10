"""
Image Retrieval with CLIP on Amazon Berkeley Objects (ABO)
──────────────────────────────────────────────────────────
Stages
  1. load_data  – parse ABO listings + images, build text descriptions
  2. embed      – fused embeddings: L2_norm(img_emb + txt_emb)
  3. index      – FAISS IndexFlatIP (exact cosine similarity)
  4. evaluate   – Precision@K, Recall@K, mAP@K, nDCG@K, MRR
  5. finetune   – symmetric contrastive loss on ABO image-text pairs
                  backbone cached → projection head only trains
  6. visualise  – matplotlib retrieval grid

Modes (set MODE below):
  "pretrained"  – embed → index → evaluate (zero-shot)
  "finetuned"   – finetune → embed → index → evaluate
"""

import os, sys, json, gzip, pickle, random, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import faiss
import matplotlib.pyplot as plt
from transformers import (CLIPProcessor,
                          CLIPVisionModelWithProjection,
                          CLIPTextModelWithProjection)

# ─── Config ────────────────────────────────────────────────────────────────────
MODE      = "finetuned"   # "pretrained" | "finetuned"

MODEL_ID  = "openai/clip-vit-base-patch32"

# DATA_ROOT = "/content/data" # fast local disk
# LOG_ROOT  = "/content/drive/MyDrive/NEU/acads/courses/sem4/cs7180/project/log" # persists on Drive

# ABO_LISTINGS_DIR = f"{DATA_ROOT}/listings/metadata"
# ABO_IMAGES_DIR   = f"{DATA_ROOT}/images/small"
# IMAGE_META_PATH  = f"{DATA_ROOT}/images/metadata/images.csv.gz"

DATA_ROOT = "./data"
LOG_ROOT  = "./log"

ABO_LISTINGS_DIR = f"{DATA_ROOT}/abo-listings/listings/metadata"
ABO_IMAGES_DIR   = f"{DATA_ROOT}/abo-images-small/images/small"
IMAGE_META_PATH  = f"{DATA_ROOT}/abo-images-small/images/metadata/images.csv.gz"

NUM_ITEMS = 999_999 # Full dataset
BATCH_SIZE   = 128   # larger batch for GPU embedding
FT_EPOCHS     = 100
FT_BATCH_SIZE = 512  # more negatives for contrastive training

# NUM_ITEMS = 10_000
# BATCH_SIZE   = 64
# FT_EPOCHS     = 50
# FT_BATCH_SIZE = 256

VAL_SPLIT    = 0.2
K_VALUES     = [1, 5, 10]
SEED         = 42
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Finetuning
FT_LR         = 1e-4
TEMPERATURE   = 0.07

# Visualisation
NUM_QUERY_PLOT = 5
TOP_K_PLOT     = 5

# ─── Path helpers ──────────────────────────────────────────────────────────────

def embed_path(mode):
    p = f"{LOG_ROOT}/{mode}/embs.pkl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def ckpt_path():
    p = f"{LOG_ROOT}/finetuned/model.pt"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def plot_path(mode):
    p = f"{LOG_ROOT}/{mode}/retrieval.png"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def log_path(mode):
    p = f"{LOG_ROOT}/{mode}/log.txt"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p

def backbone_cache_path():
    p = f"{LOG_ROOT}/backbone_cache.pkl"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


# ─── Tee: mirror stdout to log file ───────────────────────────────────────────

class Tee:
    def __init__(self, filepath):
        self._terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "w", buffering=1)

    def write(self, message):
        self._terminal.write(message)
        self._file.write(message)

    def flush(self):
        self._terminal.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    def isatty(self):
        return self._terminal.isatty()


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def get_english(field_list):
    if not field_list:
        return ""
    for e in field_list:
        if isinstance(e, dict) and e.get("language_tag", "").startswith("en"):
            return e.get("value", "")
    return field_list[0].get("value", "") if isinstance(field_list[0], dict) else ""

def get_english_list(field_list):
    if not field_list:
        return []
    return [e.get("value", "") for e in field_list
            if isinstance(e, dict) and e.get("language_tag", "").startswith("en")]

def build_text(item: dict) -> str:
    parts = []
    name = get_english(item.get("item_name", []))
    if name:
        parts.append(name)
    category = (item.get("product_type") or [{}])[0]
    cat_val = category.get("value", "") if isinstance(category, dict) else ""
    if cat_val:
        parts.append(cat_val.replace("_", " ").title())
    colors = get_english_list(item.get("color", []))
    if colors:
        parts.append("Color: " + ", ".join(colors))
    materials = get_english_list(item.get("material", []))
    if materials:
        parts.append("Material: " + ", ".join(materials))
    styles = get_english_list(item.get("style", []))
    if styles:
        parts.append("Style: " + ", ".join(styles))
    return ". ".join(p.strip() for p in parts if p.strip()) or "product"

def load_image_id_to_path(path: str) -> dict:
    id_to_path = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_path[row["image_id"]] = row["path"]
    return id_to_path

def load_abo_items(n: int) -> list[dict]:
    """
    Sample n items evenly across all listing files for broad category coverage.
    For large n (e.g. full dataset), quota is huge so all valid items are used.
    """
    print("Building image_id → path index ...")
    id_to_path = load_image_id_to_path(IMAGE_META_PATH)

    listing_files = sorted(Path(ABO_LISTINGS_DIR).glob("*.json.gz"))
    if not listing_files:
        raise FileNotFoundError(f"No .json.gz files in {ABO_LISTINGS_DIR}")
    print(f"Scanning {len(listing_files)} listing file(s) ...")

    quota     = -(-n // len(listing_files))   # ceiling division
    all_items = []

    for fpath in listing_files:
        file_items = []
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                main_images = item.get("main_image_id", [])
                if not main_images:
                    continue
                img_id   = main_images if isinstance(main_images, str) else main_images[0]
                rel_path = id_to_path.get(img_id)
                if not rel_path:
                    continue
                abs_path = os.path.join(ABO_IMAGES_DIR, rel_path)
                if not os.path.exists(abs_path):
                    continue
                category = (item.get("product_type") or [{}])[0]
                category = (category.get("value", "UNKNOWN")
                            if isinstance(category, dict) else "UNKNOWN")
                file_items.append({
                    "text":       build_text(item),
                    "image_path": abs_path,
                    "category":   category,
                    "item_id":    item.get("item_id", ""),
                })

        if len(file_items) > quota:
            file_items = random.sample(file_items, quota)
        all_items.extend(file_items)

    random.shuffle(all_items)
    all_items = all_items[:n]
    if len(all_items) < n:
        print(f"Warning: only found {len(all_items)} valid items (requested {n}).")
    return all_items


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CLIPDualEncoder(nn.Module):
    """
    CLIP dual encoder.
    Vision: CLIPVisionModelWithProjection → .image_embeds  (B, 512)
    Text:   CLIPTextModelWithProjection   → .text_embeds   (B, 512)
    """
    def __init__(self, model_id: str):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.text   = CLIPTextModelWithProjection.from_pretrained(model_id)

    def encode_image(self, pixel_values):
        return self.vision(pixel_values=pixel_values).image_embeds

    def encode_text(self, input_ids, attention_mask):
        return self.text(input_ids=input_ids,
                         attention_mask=attention_mask).text_embeds

    def forward(self, pixel_values, input_ids, attention_mask):
        img = F.normalize(self.encode_image(pixel_values), dim=-1)
        txt = F.normalize(self.encode_text(input_ids, attention_mask), dim=-1)
        return img, txt


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ABODataset(Dataset):
    def __init__(self, items: list[dict], processor: CLIPProcessor):
        self.items     = items
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        try:
            image = Image.open(it["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224))
        enc = self.processor(
            text=[it["text"]],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        return {
            "pixel_values":   enc["pixel_values"].squeeze(0),
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. EMBEDDING
# ══════════════════════════════════════════════════════════════════════════════

def compute_fused_embeddings(items, model: CLIPDualEncoder,
                              processor, desc="Embedding",
                              save_backbone_cache_for=None) -> np.ndarray:
    """
    Compute fused L2_norm(img_emb + txt_emb) embeddings.

    If save_backbone_cache_for is an index array, backbone outputs
    (pre-projection pooler_output) for those items are saved as a side
    effect — free during the pretrained pass, used to speed up finetuning.
    """
    loader = DataLoader(ABODataset(items, processor),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=(DEVICE == "cuda"))
    model.eval()
    all_embs  = []
    cache_img = [] if save_backbone_cache_for is not None else None
    cache_txt = [] if save_backbone_cache_for is not None else None

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            pv  = batch["pixel_values"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            msk = batch["attention_mask"].to(DEVICE)

            img_backbone = model.vision.vision_model(
                pixel_values=pv).pooler_output                        # (B, 768)
            txt_backbone = model.text.text_model(
                input_ids=ids, attention_mask=msk).pooler_output      # (B, 512)

            img_emb = F.normalize(
                model.vision.visual_projection(img_backbone), dim=-1) # (B, 512)
            txt_emb = F.normalize(
                model.text.text_projection(txt_backbone), dim=-1)     # (B, 512)

            fused = F.normalize(img_emb + txt_emb, dim=-1)
            all_embs.append(fused.cpu().float().numpy())

            if cache_img is not None:
                cache_img.append(img_backbone.cpu().float().numpy())
                cache_txt.append(txt_backbone.cpu().float().numpy())

    if save_backbone_cache_for is not None:
        all_img = np.concatenate(cache_img, axis=0)[save_backbone_cache_for]
        all_txt = np.concatenate(cache_txt, axis=0)[save_backbone_cache_for]
        with open(backbone_cache_path(), "wb") as f:
            pickle.dump({"img": all_img, "txt": all_txt}, f)
        print(f"Backbone cache saved to '{backbone_cache_path()}'.")

    return np.concatenate(all_embs, axis=0)


def compute_fused_embeddings_from_cache(model: CLIPDualEncoder) -> np.ndarray:
    """
    Compute fused embeddings via projection head only — no backbone forward pass.
    Uses backbone outputs cached during the pretrained run.
    """
    with open(backbone_cache_path(), "rb") as f:
        cache = pickle.load(f)
    img_b = torch.tensor(cache["img"], dtype=torch.float32)
    txt_b = torch.tensor(cache["txt"], dtype=torch.float32)

    model.eval()
    all_embs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(img_b), BATCH_SIZE),
                          desc="Embedding (cached)"):
            ib = img_b[start: start + BATCH_SIZE].to(DEVICE)
            tb = txt_b[start: start + BATCH_SIZE].to(DEVICE)
            ie = F.normalize(model.vision.visual_projection(ib), dim=-1)
            te = F.normalize(model.text.text_projection(tb), dim=-1)
            all_embs.append(
                F.normalize(ie + te, dim=-1).cpu().float().numpy()
            )
    return np.concatenate(all_embs, axis=0)


def load_or_compute_embeddings(items, model, processor, mode,
                               n_val=None) -> np.ndarray:
    path = embed_path(mode)
    if os.path.exists(path):
        print(f"Loading cached embeddings from '{path}' ...")
        with open(path, "rb") as f:
            return pickle.load(f)

    if mode == "finetuned" and os.path.exists(backbone_cache_path()):
        print("Using backbone cache for finetuned embeddings ...")
        embs = compute_fused_embeddings_from_cache(model)
    else:
        # Pretrained pass: also cache backbone outputs for ALL items
        backbone_indices = None
        if mode == "pretrained" and n_val is not None:
            if not os.path.exists(backbone_cache_path()):
                backbone_indices = np.arange(len(items))
        embs = compute_fused_embeddings(items, model, processor,
                                        save_backbone_cache_for=backbone_indices)

    with open(path, "wb") as f:
        pickle.dump(embs, f)
    print(f"Embeddings saved to '{path}'.")
    return embs


# ══════════════════════════════════════════════════════════════════════════════
# 5. FAISS INDEX
# ══════════════════════════════════════════════════════════════════════════════

def build_faiss_index(gallery_embs: np.ndarray) -> faiss.IndexFlatIP:
    dim   = gallery_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(gallery_embs.astype(np.float32))
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")
    return index

def faiss_retrieve(query_embs: np.ndarray,
                   index: faiss.IndexFlatIP, top_k: int) -> np.ndarray:
    _, indices = index.search(query_embs.astype(np.float32), top_k)
    return indices


# ══════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(ret_lbls, q_lbl, k):
    return np.sum(ret_lbls[:k] == q_lbl) / k

def recall_at_k(ret_lbls, q_lbl, k, gallery_labels):
    n_rel = np.sum(gallery_labels == q_lbl)
    return 0.0 if n_rel == 0 else np.sum(ret_lbls[:k] == q_lbl) / n_rel

def average_precision(ret_lbls, q_lbl, gallery_labels):
    """
    Strict AP — denominator is total relevant items in the full gallery.
    Penalises for relevant items not retrieved at all.
    Standard IR convention (TREC, most academic benchmarks).
    """
    n_rel = np.sum(gallery_labels == q_lbl)
    if n_rel == 0:
        return 0.0
    hits = ap = 0.0
    for rank, lbl in enumerate(ret_lbls, 1):
        if lbl == q_lbl:
            hits += 1
            ap += hits / rank
    return ap / n_rel


def average_precision_at_k(ret_lbls, q_lbl, gallery_labels):
    """
    AP@K — denominator is min(K, total relevant items in gallery).
    Only evaluates within the retrieved window.
    Gives higher numbers; common in recommendation system benchmarks.
    """
    k     = len(ret_lbls)
    n_rel = np.sum(gallery_labels == q_lbl)
    if n_rel == 0:
        return 0.0
    hits = ap = 0.0
    for rank, lbl in enumerate(ret_lbls, 1):
        if lbl == q_lbl:
            hits += 1
            ap += hits / rank
    return ap / min(k, n_rel)

def evaluate(retrieved_idx, query_labels, gallery_labels, k_values):
    """
    Fully vectorised evaluation — no Python loops over queries.
    Computes P@K, Recall@K, strict mAP, and mAP@K for each K.
    """
    max_k = max(k_values)
    n_q   = len(query_labels)

    # ret_matrix: (Q, max_k) — category label of each retrieved item
    ret_matrix = gallery_labels[retrieved_idx[:, :max_k]]

    # hits: (Q, max_k) — 1 if retrieved item matches query category
    hits = (ret_matrix == query_labels[:, None]).astype(np.float32)

    # n_rel: (Q,) — total relevant items in gallery per query
    n_rel = np.array([np.sum(gallery_labels == q) for q in query_labels],
                     dtype=np.float32)
    # avoid divide-by-zero: replace 0 with 1 for division, mask result after
    safe_n_rel = np.where(n_rel > 0, n_rel, 1.0)

    results = {}

    for k in k_values:
        # Precision@K
        results[f"P@{k}"] = float(hits[:, :k].sum(axis=1).mean() / k)

        # Recall@K
        rel_in_k = hits[:, :k].sum(axis=1)
        recall   = np.where(n_rel > 0, rel_in_k / safe_n_rel, 0.0)
        results[f"R@{k}"] = float(recall.mean())

    # Cumulative precision at each rank position
    cumhits     = np.cumsum(hits, axis=1)               # (Q, max_k)
    ranks       = np.arange(1, max_k + 1, dtype=np.float32)
    prec_at_hit = (cumhits / ranks) * hits              # (Q, max_k)

    # mAP@K for each K — denominator = min(K, n_rel)
    for k in k_values:
        sum_prec_k = prec_at_hit[:, :k].sum(axis=1)
        ap_at_k    = np.where(n_rel > 0,
                              sum_prec_k / np.minimum(k, safe_n_rel),
                              0.0)
        results[f"mAP@{k}"] = float(ap_at_k.mean())

    # nDCG@K for each K
    # DCG@K = sum(hits / log2(rank+1)) for rank in 1..K
    # IDCG@K = sum(1 / log2(rank+1)) for rank in 1..min(K, n_rel)  (ideal)
    log_ranks = np.log2(np.arange(2, max_k + 2, dtype=np.float32))  # log2(2..K+1)
    discounts = hits / log_ranks                                      # (Q, max_k)

    for k in k_values:
        dcg  = discounts[:, :k].sum(axis=1)                          # (Q,)
        ideal_ranks = np.arange(1, k + 1, dtype=np.float32)
        ideal_disc  = 1.0 / np.log2(ideal_ranks + 1)                 # (k,)
        idcg = np.array([
            ideal_disc[:min(int(nr), k)].sum() for nr in n_rel
        ], dtype=np.float32)
        safe_idcg = np.where(idcg > 0, idcg, 1.0)
        ndcg = np.where(idcg > 0, dcg / safe_idcg, 0.0)
        results[f"nDCG@{k}"] = float(ndcg.mean())

    # MRR — reciprocal rank of first correct result within top max_k
    first_hit_ranks = np.argmax(hits > 0, axis=1).astype(np.float32) + 1  # 1-based
    has_hit         = hits.any(axis=1)
    rr              = np.where(has_hit, 1.0 / first_hit_ranks, 0.0)
    results["MRR"]  = float(rr.mean())

    return results


def print_metrics(metrics, k_values):
    c0, c1, c2, c3, c4 = 4, 15, 15, 12, 12
    header = (f"  {'K':<{c0}} | {'Precision@K':>{c1}} | "
              f"{'Recall@K':>{c2}} | {'mAP@K':>{c3}} | {'nDCG@K':>{c4}}")
    sep    = "  " + "-" * (c0 + 1) + "+" + "-" * (c1 + 2) + "+" + \
             "-" * (c2 + 2) + "+" + "-" * (c3 + 2) + "+" + "-" * (c4 + 1)
    print("\n── Retrieval Metrics ───────────────────────────────────────────────────")
    print(header)
    print(sep)
    for k in k_values:
        print(f"  {k:<{c0}} | {metrics[f'P@{k}']:>{c1}.4f} | "
              f"{metrics[f'R@{k}']:>{c2}.4f} | {metrics[f'mAP@{k}']:>{c3}.4f} | "
              f"{metrics[f'nDCG@{k}']:>{c4}.4f}")
    print(sep)
    print(f"  MRR         : {metrics['MRR']:.4f}")
    print("────────────────────────────────────────────────────────────────────────")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FINETUNING
# ══════════════════════════════════════════════════════════════════════════════

class ContrastiveLoss(nn.Module):
    """
    Symmetric cross-entropy over B×B cosine similarity matrix.
    Diagonal = positive pairs (same item image + text).
    Off-diagonal = implicit in-batch negatives.
    """
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb, txt_emb):
        logits = (img_emb @ txt_emb.T) / self.temperature
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2


class BackboneCacheDataset(Dataset):
    """Serves pre-projection backbone outputs for projection-head-only training."""
    def __init__(self, img_feats: np.ndarray, txt_feats: np.ndarray):
        self.img = torch.tensor(img_feats, dtype=torch.float32)
        self.txt = torch.tensor(txt_feats, dtype=torch.float32)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.txt[idx]


def finetune_model(gallery_items, model: CLIPDualEncoder, processor):
    """
    Finetune CLIP projection heads on gallery items.
    Backbone is frozen; backbone outputs are loaded from cache.
    Each epoch runs in seconds since no backbone forward pass is needed.
    """
    print(f"\nFinetuning CLIP for {FT_EPOCHS} epoch(s) "
          f"on {len(gallery_items)} gallery items ...")

    # Load backbone cache (built during pretrained run)
    with open(backbone_cache_path(), "rb") as f:
        cache = pickle.load(f)

    # Cache has ALL items (queries + gallery); slice to gallery only.
    # n_val = total cached items - gallery size
    n_val       = len(cache["img"]) - len(gallery_items)
    gallery_img = cache["img"][n_val:]
    gallery_txt = cache["txt"][n_val:]

    # Freeze backbone, keep only projection heads trainable
    for p in model.parameters():
        p.requires_grad = False
    for p in (list(model.vision.visual_projection.parameters()) +
              list(model.text.text_projection.parameters())):
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    total     = sum(p.numel() for p in model.parameters())
    n_train   = sum(p.numel() for p in trainable)
    print(f"  Trainable: {n_train:,} / {total:,} ({100 * n_train / total:.1f}%)")

    # Both gallery and query backbone outputs are in the cache.
    # Slice accordingly: cache is ordered [queries | gallery]
    query_img = cache["img"][:n_val]
    query_txt = cache["txt"][:n_val]

    loader = DataLoader(BackboneCacheDataset(gallery_img, gallery_txt),
                        batch_size=FT_BATCH_SIZE, shuffle=True, drop_last=True)

    # Validation loader uses cached query backbone outputs — no forward pass needed
    val_loader = DataLoader(BackboneCacheDataset(query_img, query_txt),
                            batch_size=FT_BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(trainable, lr=FT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FT_EPOCHS * len(loader))
    criterion = ContrastiveLoss().to(DEVICE)

    train_losses, val_losses = [], []

    model.train()
    for epoch in range(1, FT_EPOCHS + 1):
        epoch_loss = 0.0
        for img_b, txt_b in loader:
            img_b = img_b.to(DEVICE)
            txt_b = txt_b.to(DEVICE)
            ie = F.normalize(model.vision.visual_projection(img_b), dim=-1)
            te = F.normalize(model.text.text_projection(txt_b), dim=-1)
            loss = criterion(ie, te)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(loader)
        train_losses.append(train_loss)

        # Validation loss every epoch (fast — projection head + cached inputs)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img_b, txt_b in val_loader:
                img_b = img_b.to(DEVICE)
                txt_b = txt_b.to(DEVICE)
                ie = F.normalize(model.vision.visual_projection(img_b), dim=-1)
                te = F.normalize(model.text.text_projection(txt_b), dim=-1)
                val_loss += criterion(ie, te).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        model.train()

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} — train loss: {train_loss:.4f}"
                  f"  val loss: {val_loss:.4f}")

    # Plot loss curves
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, FT_EPOCHS + 1)
    ax.plot(epochs, train_losses, label="Train (gallery)")
    ax.plot(epochs, val_losses,   label="Val (queries)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("CLIP Finetuning — Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_plot = f"{LOG_ROOT}/finetuned/loss_curves.png"
    os.makedirs(os.path.dirname(loss_plot), exist_ok=True)
    plt.savefig(loss_plot, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Loss curves saved to '{loss_plot}'.")

    torch.save(model.state_dict(), ckpt_path())
    print(f"Checkpoint saved to '{ckpt_path()}'.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_retrievals(query_items, gallery_items,
                    retrieved_idx, query_labels, gallery_labels, mode):
    sample_qi = np.random.choice(len(query_labels), NUM_QUERY_PLOT, replace=False)
    fig, axes = plt.subplots(NUM_QUERY_PLOT, TOP_K_PLOT + 1,
                             figsize=(2.6 * (TOP_K_PLOT + 1), 3.0 * NUM_QUERY_PLOT))
    for row, qi in enumerate(sample_qi):
        ax = axes[row, 0]
        try:
            ax.imshow(Image.open(query_items[qi]["image_path"]).convert("RGB"))
        except Exception:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"QUERY\n{query_items[qi]['category'].replace('_',' ').title()[:20]}",
                     fontsize=7, color="steelblue", fontweight="bold")
        ax.axis("off")
        for col, gidx in enumerate(retrieved_idx[qi][:TOP_K_PLOT], 1):
            ax = axes[row, col]
            try:
                ax.imshow(Image.open(gallery_items[gidx]["image_path"]).convert("RGB"))
            except Exception:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            correct = gallery_labels[gidx] == query_labels[qi]
            ax.set_title(f"#{col} {'✓' if correct else '✗'}\n"
                         f"{gallery_items[gidx]['category'].replace('_',' ').title()[:20]}",
                         fontsize=7, color="green" if correct else "red")
            ax.axis("off")
    label = "Pretrained" if mode == "pretrained" else "Finetuned"
    plt.suptitle(f"ABO Retrieval — CLIP {label} (fused image+text embeddings)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(plot_path(mode), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved to '{plot_path(mode)}'.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_eval(items, query_items, gallery_items,
             query_labels, gallery_labels, model, processor, mode):
    n_val        = len(query_items)
    all_embs     = load_or_compute_embeddings(items, model, processor,
                                              mode, n_val=n_val)
    query_embs   = all_embs[:n_val]
    gallery_embs = all_embs[n_val:]
    print(f"  Query embs: {query_embs.shape} | Gallery embs: {gallery_embs.shape}")
    index         = build_faiss_index(gallery_embs)
    retrieved_idx = faiss_retrieve(query_embs, index, top_k=max(K_VALUES))
    metrics       = evaluate(retrieved_idx, query_labels, gallery_labels, K_VALUES)
    print_metrics(metrics, K_VALUES)
    plot_retrievals(query_items, gallery_items,
                    retrieved_idx, query_labels, gallery_labels, mode)


def main():
    tee = Tee(log_path(MODE))
    sys.stdout = tee
    try:
        print(f"Device  : {DEVICE}")
        print(f"Model   : {MODEL_ID}")
        print(f"Mode    : {MODE}\n")

        # 1. Data
        print("[1] Loading ABO items ...")
        items = load_abo_items(NUM_ITEMS)
        print(f"  Total: {len(items)} items")
        random.shuffle(items)
        n_val          = int(len(items) * VAL_SPLIT)
        query_items    = items[:n_val]
        gallery_items  = items[n_val:]
        query_labels   = np.array([it["category"] for it in query_items])
        gallery_labels = np.array([it["category"] for it in gallery_items])
        n_cats = len(set(query_labels) | set(gallery_labels))
        print(f"  Gallery: {len(gallery_items)} | Queries: {len(query_items)} | "
              f"Categories: {n_cats}")

        # 2. Model (always load — debug now needs it too)
        print(f"\n[2] Loading {MODEL_ID} ...")
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        model     = CLIPDualEncoder(MODEL_ID).to(DEVICE)

        # 3. Run
        if MODE == "debug":
            print("\n[3] Debug — inspecting a random item ...")
            item = random.choice(items)
            print(f"  item_id    : {item['item_id']}")
            print(f"  category   : {item['category']}")
            print(f"  text       : {item['text']}")
            print(f"  image_path : {item['image_path']}")
            try:
                img = Image.open(item["image_path"]).convert("RGB")
                print(f"  image size : {img.size}  (W x H)")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(img)
                ax.set_title(f"{item['category'].replace('_',' ').title()}\n"
                             f"{item['text'][:80]}{'...' if len(item['text']) > 80 else ''}",
                             fontsize=8, wrap=True)
                ax.axis("off")
                plt.tight_layout()
                out = f"{LOG_ROOT}/debug_item.png"
                os.makedirs(LOG_ROOT, exist_ok=True)
                plt.savefig(out, dpi=150, bbox_inches="tight")
                plt.show()
                print(f"  Image saved to '{out}'.")
            except Exception as e:
                print(f"  Could not load image: {e}")

            # ── Pass item through model, print shapes and norms ──
            print("\n[4] Debug — model outputs ...")
            enc = processor(
                text=[item["text"]],
                images=[img],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            )
            pv  = enc["pixel_values"].to(DEVICE)
            ids = enc["input_ids"].to(DEVICE)
            msk = enc["attention_mask"].to(DEVICE)

            model.eval()
            with torch.no_grad():
                vision_backbone = model.vision.vision_model(
                    pixel_values=pv).pooler_output               # (1, 768)
                text_backbone   = model.text.text_model(
                    input_ids=ids,
                    attention_mask=msk).pooler_output            # (1, 512)
                vision_proj     = model.vision.visual_projection(
                    vision_backbone)                             # (1, 512)
                text_proj       = model.text.text_projection(
                    text_backbone)                               # (1, 512)

            def fmt(name, t):
                t = t.squeeze(0)   # remove batch dim for display
                print(f"  {name:<22} shape: {tuple(t.shape)}  "
                      f"norm: {t.norm().item():.4f}")

            fmt("vision backbone",   vision_backbone)
            fmt("text backbone",     text_backbone)
            fmt("vision projection", vision_proj)
            fmt("text projection",   text_proj)
            fmt("vision proj (L2)",  F.normalize(vision_proj, dim=-1))
            fmt("text proj (L2)",    F.normalize(text_proj,   dim=-1))

            # ── Metric verification ────────────────────────────────────────
            print("\n[5] Debug — metric verification ...")

            # Compute query embedding for the debug item
            with torch.no_grad():
                img_bb  = model.vision.vision_model(pixel_values=pv).pooler_output
                txt_bb  = model.text.text_model(input_ids=ids, attention_mask=msk).pooler_output
                ie      = F.normalize(model.vision.visual_projection(img_bb), dim=-1)
                te      = F.normalize(model.text.text_projection(txt_bb), dim=-1)
                query_emb = F.normalize(ie + te, dim=-1).cpu().numpy()  # (1, 512)

            embs_file = embed_path("pretrained")
            if not os.path.exists(embs_file):
                embs_file = embed_path("finetuned")
            if not os.path.exists(embs_file):
                print("  No saved embeddings found. Run pretrained or finetuned mode first.")
            else:
                print(f"  Loading embeddings from '{embs_file}' ...")
                with open(embs_file, "rb") as f:
                    all_embs = pickle.load(f)
                gallery_embs_np = all_embs[n_val:]

                # Retrieve top-K for this item
                index = build_faiss_index(gallery_embs_np)
                top_k = max(K_VALUES)
                indices = faiss_retrieve(query_emb, index, top_k=top_k).squeeze(0)
                ret_lbls = gallery_labels[indices]

                print(f"  Query category : {item['category']}")
                print(f"  Top-{top_k} retrieved categories:")
                for rank, (idx, lbl) in enumerate(zip(indices, ret_lbls), 1):
                    match = "✓" if lbl == item["category"] else "✗"
                    print(f"    #{rank:2d} {match} {lbl}")

                # Metrics for this single item
                for k in K_VALUES:
                    p = precision_at_k(ret_lbls, item["category"], k)
                    r = recall_at_k(ret_lbls, item["category"], k, gallery_labels)
                    print(f"  P@{k}: {p:.4f}  R@{k}: {r:.4f}")
                ap = average_precision_at_k(gallery_labels[indices], item["category"], gallery_labels)
                print(f"  AP  : {ap:.4f}")

        elif MODE == "pretrained":
            print("\n[3] Zero-shot evaluation ...")
            run_eval(items, query_items, gallery_items,
                     query_labels, gallery_labels,
                     model, processor, mode="pretrained")

        elif MODE == "finetuned":
            _ckpt = ckpt_path()
            if os.path.exists(_ckpt):
                print(f"\n[3] Loading checkpoint '{_ckpt}' ...")
                model.load_state_dict(torch.load(_ckpt, map_location=DEVICE,
                                                 weights_only=True))
            else:
                model = finetune_model(gallery_items, model, processor)
            print("\n[4] Evaluating finetuned model ...")
            run_eval(items, query_items, gallery_items,
                     query_labels, gallery_labels,
                     model, processor, mode="finetuned")

        else:
            raise ValueError(f"Unknown MODE '{MODE}'. Use 'pretrained', 'finetuned', or 'debug'.")

    finally:
        sys.stdout = tee._terminal
        tee.close()
        print(f"Log saved to '{log_path(MODE)}'.")


if __name__ == "__main__":
    main()