"""
Multimodal Semantic Product Search — Streamlit UI
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import json
from PIL import Image
from torchvision import transforms
from main import Config, TwoTowerModel, ABODataset, query_index, parse_attributes

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


def attribute_rerank(results, query_attrs, item_lookup, fusion_weight=0.6):
    """
    Re-rank results by combining the fusion similarity score with
    an attribute match score. This boosts items that share specific
    attributes (color, material, brand, etc.) with the query.
    """
    # Parse query attributes into a dict: {"color": "white", "product_type": "couch"}
    query_dict = {}
    for attr in query_attrs:
        if ":" in attr:
            key, val = attr.split(":", 1)
            query_dict[key.strip().lower()] = val.strip().lower()

    reranked = []
    for score, pid in results:
        info = item_lookup.get(pid)
        attr_score = 0.0
        if info and query_dict:
            catalog_attrs = info["attributes"].split(" | ")
            catalog_dict = {}
            for a in catalog_attrs:
                if ":" in a:
                    k, v = a.split(":", 1)
                    catalog_dict[k.strip().lower()] = v.strip().lower()

            # Count matching attributes
            matches = 0
            for key, val in query_dict.items():
                if key in catalog_dict:
                    # Exact match
                    if val == catalog_dict[key].lower():
                        matches += 1
                    # Partial/substring match
                    elif val in catalog_dict[key].lower() or catalog_dict[key].lower() in val:
                        matches += 0.5

            attr_score = matches / len(query_dict) if query_dict else 0.0

        # Combine: fusion similarity + attribute match bonus
        combined = fusion_weight * score + (1 - fusion_weight) * attr_score
        reranked.append((combined, score, attr_score, pid))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


cfg, model, catalog_embs, catalog_ids, item_lookup = load_resources()

with st.sidebar:
    st.header("Query Inputs")
    uploaded = st.file_uploader("Product Image", type=["jpg", "jpeg", "png"])
    brand = st.text_input("Brand", placeholder="e.g. Rivet")
    material = st.text_input("Material", placeholder="e.g. Oak")
    color = st.text_input("Color", placeholder="e.g. Natural")
    ptype = st.text_input("Product Type", placeholder="e.g. Table")
    top_k = st.slider("Top-K results", 1, 20, 5)

    st.divider()
    st.subheader("Advanced")
    rerank_weight = st.slider(
        "Attribute re-ranking strength",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Lower = rely more on attribute matching. Higher = rely more on neural similarity."
    )

if uploaded is not None:
    col_query, col_spacer = st.columns([1, 2])
    with col_query:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", use_container_width=True)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_t = tfm(img).unsqueeze(0).to(cfg.device)

    # Build attribute list from user inputs
    parts = []
    if brand:    parts.append(f"brand: {brand}")
    if material: parts.append(f"material: {material}")
    if color:    parts.append(f"color: {color}")
    if ptype:    parts.append(f"product_type: {ptype}")
    query_text = " | ".join(parts) if parts else "unknown product"

    st.markdown(f"**Attribute query:** `{query_text}`")

    if st.button("🔎 Search"):
        query_attrs = parse_attributes(query_text)

        with torch.no_grad():
            emb = model(img_t, [query_attrs]).cpu().numpy().astype("float32")

        # Stage 1: Get broad candidates using neural fusion (fetch extra for re-ranking)
        candidate_k = min(top_k * 10, len(catalog_ids))
        raw_results = query_index(emb, catalog_embs, catalog_ids, k=candidate_k)[0]

        # Stage 2: Re-rank candidates by attribute matching
        reranked = attribute_rerank(
            raw_results, query_attrs, item_lookup, fusion_weight=rerank_weight
        )

        st.subheader(f"Top-{top_k} Results")

        # Display results in a grid of columns
        cols = st.columns(min(top_k, 5))
        for i, (combined, sim_score, attr_score, pid) in enumerate(reranked[:top_k]):
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

                st.markdown(f"**#{i+1}** — `{combined:.3f}`")
                st.caption(f"sim: {sim_score:.3f} | attr: {attr_score:.3f}")
                st.caption(f"ID: {pid}")

                if info:
                    attrs = info["attributes"].split(" | ")
                    for a in attrs[:4]:
                        st.caption(a)