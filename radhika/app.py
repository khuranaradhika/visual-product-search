"""
Multimodal Semantic Product Search — Streamlit UI
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import torch
import json
from PIL import Image
from torchvision import transforms
from main import Config, TwoTowerModel, ABODataset, query_index

st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("🔍 Multimodal Semantic Product Search")
st.markdown("Upload an image and specify attributes to find matching products.")


@st.cache_resource
def load_resources():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()
    embs_path = cfg.index_save_path.replace(".bin", "_embs.npy")
    catalog_embs = np.load(embs_path)
    with open(cfg.catalog_ids_path) as f:
        catalog_ids = json.load(f)

    # Build item_id → (image_path, attributes) lookup from dataset
    ds = ABODataset(cfg)
    item_lookup = {}
    for img_path, attr_text, item_id in ds.samples:
        item_lookup[item_id] = {"image_path": img_path, "attributes": attr_text}

    return cfg, model, catalog_embs, catalog_ids, item_lookup


cfg, model, catalog_embs, catalog_ids, item_lookup = load_resources()

with st.sidebar:
    st.header("Query Inputs")
    uploaded = st.file_uploader("Product Image", type=["jpg", "jpeg", "png"])
    brand = st.text_input("Brand", placeholder="e.g. Rivet")
    material = st.text_input("Material", placeholder="e.g. Oak")
    color = st.text_input("Color", placeholder="e.g. Natural")
    ptype = st.text_input("Product Type", placeholder="e.g. Table")
    top_k = st.slider("Top-K results", 1, 20, 5)

if uploaded is not None:
    col_query, col_spacer = st.columns([1, 2])
    with col_query:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", use_container_width=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t = tfm(img).unsqueeze(0).to(cfg.device)

    parts = []
    if brand:    parts.append(f"brand: {brand}")
    if material: parts.append(f"material: {material}")
    if color:    parts.append(f"color: {color}")
    if ptype:    parts.append(f"product_type: {ptype}")
    query_text = " | ".join(parts) if parts else "unknown product"

    st.markdown(f"**Attribute query:** `{query_text}`")

    if st.button("🔎 Search"):
        with torch.no_grad():
            emb = model(img_t, [query_text]).cpu().numpy().astype("float32")
        results = query_index(emb, catalog_embs, catalog_ids, k=top_k)[0]

        st.subheader(f"Top-{top_k} Results")
        # Display results in a grid of columns
        cols = st.columns(min(top_k, 5))
        for i, (score, pid) in enumerate(results):
            with cols[i % len(cols)]:
                info = item_lookup.get(pid)
                if info and info["image_path"]:
                    try:
                        result_img = Image.open(info["image_path"]).convert("RGB")
                        st.image(result_img, use_container_width=True)
                    except Exception:
                        st.write("⚠️ Image not found")
                else:
                    st.write("⚠️ Image not found")

                st.markdown(f"**#{i+1}** — `{score:.3f}`")
                st.caption(f"ID: {pid}")
                if info:
                    # Show attributes in a compact way
                    attrs = info["attributes"].split(" | ")
                    for a in attrs[:4]:  # show up to 4 attributes
                        st.caption(a)