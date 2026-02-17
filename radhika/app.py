"""
Multimodal Semantic Product Search — Streamlit UI
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import torch
import json
import faiss
from PIL import Image
from torchvision import transforms
from main import Config, TwoTowerModel, query_index

st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("Multimodal Semantic Product Search")


@st.cache_resource
def load_resources():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()
    index = faiss.read_index(cfg.index_save_path)
    with open(cfg.catalog_ids_path) as f:
        catalog_ids = json.load(f)
    return cfg, model, index, catalog_ids


cfg, model, index, catalog_ids = load_resources()

with st.sidebar:
    st.header("Query Inputs")
    uploaded = st.file_uploader("Product Image", type=["jpg", "jpeg", "png"])
    brand = st.text_input("Brand", placeholder="e.g. Rivet")
    material = st.text_input("Material", placeholder="e.g. Oak")
    color = st.text_input("Color", placeholder="e.g. Natural")
    ptype = st.text_input("Product Type", placeholder="e.g. Table")
    top_k = st.slider("Top-K results", 1, 20, 5)

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Query image", width=250)

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

    if st.button("Search"):
        with torch.no_grad():
            emb = model(img_t, [query_text]).cpu().numpy().astype("float32")
        results = query_index(emb, index, catalog_ids, k=top_k)[0]

        st.subheader(f"Top-{top_k} Results")
        for i, (score, pid) in enumerate(results):
            st.write(f"**#{i+1}** — score: {score:.3f} — item_id: `{pid}`")