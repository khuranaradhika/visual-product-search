"""
Multimodal Semantic Product Search — Streamlit UI (CLIP)
Run with:  streamlit run app.py
"""

import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from transformers import (CLIPProcessor,
                          CLIPVisionModelWithProjection,
                          CLIPTextModelWithProjection)

# ─── Config (must match model_colab.py) ────────────────────────────────────────
MODEL_ID  = "openai/clip-vit-base-patch32"
LOG_ROOT  = "./log"

EMBED_PATHS = {
    "Pretrained": f"{LOG_ROOT}/pretrained/embs.pkl",
    "Finetuned":  f"{LOG_ROOT}/finetuned/embs.pkl",
}
CKPT_PATH = f"{LOG_ROOT}/finetuned/model.pt"

ABO_LISTINGS_DIR = "./data/abo-listings/listings/metadata"
ABO_IMAGES_DIR   = "./data/abo-images-small/images/small"
IMAGE_META_PATH  = "./data/abo-images-small/images/metadata/images.csv.gz"
NUM_ITEMS        = 999_999   # must match model_colab.py
VAL_SPLIT        = 0.2       # must match model_colab.py
SEED             = 42        # must match model_colab.py
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Model ─────────────────────────────────────────────────────────────────────

class CLIPDualEncoder(torch.nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        self.vision = CLIPVisionModelWithProjection.from_pretrained(model_id)
        self.text   = CLIPTextModelWithProjection.from_pretrained(model_id)

    def encode_image(self, pixel_values):
        return self.vision(pixel_values=pixel_values).image_embeds

    def encode_text(self, input_ids, attention_mask):
        return self.text(input_ids=input_ids,
                         attention_mask=attention_mask).text_embeds


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_abo_items_for_app():
    import gzip, json, csv, random
    from pathlib import Path

    def get_english(fl):
        if not fl: return ""
        for e in fl:
            if isinstance(e, dict) and e.get("language_tag", "").startswith("en"):
                return e.get("value", "")
        return fl[0].get("value", "") if isinstance(fl[0], dict) else ""

    def get_english_list(fl):
        if not fl: return []
        return [e.get("value", "") for e in fl
                if isinstance(e, dict) and e.get("language_tag", "").startswith("en")]

    def build_text(item):
        parts = []
        name = get_english(item.get("item_name", []))
        if name: parts.append(name)
        cat = (item.get("product_type") or [{}])[0]
        cv = cat.get("value", "") if isinstance(cat, dict) else ""
        if cv: parts.append(cv.replace("_", " ").title())
        colors = get_english_list(item.get("color", []))
        if colors: parts.append("Color: " + ", ".join(colors))
        mats = get_english_list(item.get("material", []))
        if mats: parts.append("Material: " + ", ".join(mats))
        styles = get_english_list(item.get("style", []))
        if styles: parts.append("Style: " + ", ".join(styles))
        return ". ".join(p.strip() for p in parts if p.strip()) or "product"

    id_to_path = {}
    with gzip.open(IMAGE_META_PATH, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            id_to_path[row["image_id"]] = row["path"]

    listing_files = sorted(Path(ABO_LISTINGS_DIR).glob("*.json.gz"))
    quota     = -(-NUM_ITEMS // len(listing_files))
    all_items = []

    for fpath in listing_files:
        file_items = []
        with gzip.open(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: item = json.loads(line)
                except: continue
                main_images = item.get("main_image_id", [])
                if not main_images: continue
                img_id = main_images if isinstance(main_images, str) else main_images[0]
                rel = id_to_path.get(img_id)
                if not rel: continue
                abs_path = os.path.join(ABO_IMAGES_DIR, rel)
                if not os.path.exists(abs_path): continue
                cat = (item.get("product_type") or [{}])[0]
                cat = cat.get("value", "UNKNOWN") if isinstance(cat, dict) else "UNKNOWN"
                file_items.append({
                    "text":       build_text(item),
                    "image_path": abs_path,
                    "category":   cat,
                    "item_id":    item.get("item_id", ""),
                })
        if len(file_items) > quota:
            file_items = random.sample(file_items, quota)
        all_items.extend(file_items)

    random.shuffle(all_items)
    return all_items[:NUM_ITEMS]


# ─── Resource loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data and models...")
def load_all_resources():
    import random
    random.seed(SEED)
    np.random.seed(SEED)

    items = load_abo_items_for_app()
    random.shuffle(items)
    n_val         = int(len(items) * VAL_SPLIT)
    gallery_items = items[n_val:]

    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    pretrained_model = CLIPDualEncoder(MODEL_ID).to(DEVICE)
    pretrained_model.eval()

    finetuned_model = CLIPDualEncoder(MODEL_ID).to(DEVICE)
    if os.path.exists(CKPT_PATH):
        finetuned_model.load_state_dict(
            torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
        )
    finetuned_model.eval()

    gallery_embs = {}
    for mode, path in EMBED_PATHS.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                all_embs = pickle.load(f)
            gallery_embs[mode] = all_embs[n_val:].astype(np.float32)

    return processor, pretrained_model, finetuned_model, gallery_items, gallery_embs


# ─── Inference ─────────────────────────────────────────────────────────────────

def embed_query(model: CLIPDualEncoder, processor, image=None, text=None):
    enc = processor(
        text=[text] if text else ["product"],
        images=[image] if image else [Image.new("RGB", (224, 224))],
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
        if image is not None and text:
            emb = F.normalize(
                F.normalize(model.encode_image(pv), dim=-1) +
                F.normalize(model.encode_text(ids, msk), dim=-1),
                dim=-1
            )
        elif image is not None:
            emb = F.normalize(model.encode_image(pv), dim=-1)
        else:
            emb = F.normalize(model.encode_text(ids, msk), dim=-1)
    return emb.cpu().numpy()


def retrieve(query_emb, gallery_embs, k):
    sim  = (torch.tensor(query_emb) @ torch.tensor(gallery_embs).T).squeeze(0)
    topk = sim.topk(k)
    return topk.indices.numpy(), topk.values.numpy()


# ─── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("🔍 Multimodal Semantic Product Search")
st.caption("Upload an image, describe a product in text, or both.")

processor, pretrained_model, finetuned_model, gallery_items, gallery_embs = \
    load_all_resources()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    ft_choice = st.radio("Weights", ["Pretrained", "Finetuned"], horizontal=True)
    top_k     = st.slider("Top-K results", 1, 20, 5)

    if ft_choice not in gallery_embs:
        st.warning(
            f"No {ft_choice.lower()} embeddings found. "
            f"Run model_colab.py with MODE='{ft_choice.lower()}' first."
        )

    st.divider()
    st.header("🖼️ Image Query")
    uploaded = st.file_uploader("Upload product image", type=["jpg", "jpeg", "png"])

    st.header("📝 Text Query")
    freeform = st.text_area("Describe the product",
                            placeholder="e.g. brown wooden rustic bar stool",
                            height=80)

    embs_available = ft_choice in gallery_embs
    search_btn = st.button(
        "🔎 Search", use_container_width=True,
        disabled=(not embs_available or
                  (uploaded is None and not freeform.strip()))
    )

# ── Query construction ────────────────────────────────────────────────────────
query_image = Image.open(uploaded).convert("RGB") if uploaded else None
query_text  = freeform.strip() if freeform.strip() else None

if query_image or query_text:
    st.subheader("Your Query")
    qc1, qc2 = st.columns([1, 2])
    with qc1:
        if query_image:
            st.image(query_image, caption="Query image", use_container_width=True)
    with qc2:
        if query_text:
            st.markdown("**Text:**")
            st.write(query_text)
        mode_label = ("Image + Text" if query_image and query_text
                      else "Image only" if query_image else "Text only")
        st.info(f"Query mode: **{mode_label}** · Weights: **{ft_choice}**")

# ── Search ────────────────────────────────────────────────────────────────────
if search_btn:
    model = pretrained_model if ft_choice == "Pretrained" else finetuned_model
    embs  = gallery_embs[ft_choice]

    with st.spinner("Embedding query..."):
        q_emb = embed_query(model, processor, image=query_image, text=query_text)

    with st.spinner("Searching gallery..."):
        indices, scores = retrieve(q_emb, embs, k=top_k)

    st.subheader(f"Top-{top_k} Results")
    cols = st.columns(min(top_k, 5))
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        item = gallery_items[idx]
        with cols[rank % len(cols)]:
            try:
                st.image(Image.open(item["image_path"]).convert("RGB"),
                         use_container_width=True)
            except Exception:
                st.warning("Image not found")
            st.markdown(f"**#{rank+1}** · score: `{score:.3f}`")
            st.caption(item["category"].replace("_", " ").title())
            for line in item["text"].split(". ")[:3]:
                st.caption(line)