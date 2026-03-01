"""
Multimodal Semantic Product Search — Streamlit UI
Run with:  streamlit run app.py
"""
import streamlit as st
import numpy as np
import torch
import json
import faiss
import os
from PIL import Image
from torchvision import transforms
from main import (
    Config, TwoTowerModel, ABODataset,
    load_faiss_index, query_index, parse_attributes,
)

st.set_page_config(page_title="Multimodal Product Search", layout="wide")
st.title("🔍 Multimodal Semantic Product Search")
st.markdown("Upload an image and optionally specify attributes to find matching products.")


@st.cache_resource
def load_resources():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    fused_index, catalog_ids = load_faiss_index(cfg)
    if hasattr(fused_index, "nprobe"):
        fused_index.nprobe = min(128, fused_index.nlist)

    img_index_path = cfg.index_save_path.replace(".bin", "_imgonly.bin")
    img_ids_path = cfg.catalog_ids_path.replace(".json", "_imgonly.json")
    if os.path.exists(img_index_path) and os.path.exists(img_ids_path):
        img_index = faiss.read_index(img_index_path)
        with open(img_ids_path) as f:
            img_catalog_ids = json.load(f)
        if hasattr(img_index, "nprobe"):
            img_index.nprobe = min(128, img_index.nlist)
    else:
        img_index = None
        img_catalog_ids = None

    ds = ABODataset(cfg)
    item_lookup = {}
    for img_path, attr_text, item_id in ds.samples:
        item_lookup[item_id] = {"image_path": img_path, "attributes": attr_text}

    return cfg, model, fused_index, catalog_ids, img_index, img_catalog_ids, item_lookup


def attribute_rerank(results, query_attrs, item_lookup, fusion_weight=0.6):
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
            catalog_dict = {}
            for a in info["attributes"].split(" | "):
                if ":" in a:
                    k, v = a.split(":", 1)
                    catalog_dict[k.strip().lower()] = v.strip().lower()
            matches = 0
            for key, val in query_dict.items():
                if key in catalog_dict:
                    if val == catalog_dict[key].lower():
                        matches += 1
                    elif (val in catalog_dict[key].lower()
                          or catalog_dict[key].lower() in val):
                        matches += 0.5
            attr_score = matches / len(query_dict) if query_dict else 0.0
        combined = fusion_weight * score + (1 - fusion_weight) * attr_score
        reranked.append((combined, score, attr_score, pid))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked


def merge_results(fused_results, img_results, img_weight=0.4):
    scores = {}
    for score, pid in fused_results:
        scores.setdefault(pid, {"fused": 0.0, "img": 0.0})
        scores[pid]["fused"] = max(scores[pid]["fused"], score)
    for score, pid in img_results:
        scores.setdefault(pid, {"fused": 0.0, "img": 0.0})
        scores[pid]["img"] = max(scores[pid]["img"], score)
    merged = []
    for pid, s in scores.items():
        combined = (1 - img_weight) * s["fused"] + img_weight * s["img"]
        if s["fused"] > 0 and s["img"] > 0:
            combined += 0.05
        merged.append((combined, pid))
    merged.sort(key=lambda x: x[0], reverse=True)
    return merged


def deduplicate_results(reranked, item_lookup):
    """Remove near-duplicate results (same product in different sizes)."""
    seen = set()
    deduped = []
    for entry in reranked:
        combined, sim_score, attr_score, pid = entry
        info = item_lookup.get(pid)
        if info:
            brand = ""
            ptype = ""
            for a in info["attributes"].split(" | "):
                k, _, v = a.partition(":")
                k = k.strip().lower()
                if k == "brand":
                    brand = v.strip().lower()
                elif k == "product_type":
                    ptype = v.strip().lower()
            score_key = f"{sim_score:.3f}"
            dedup_key = f"{brand}|{ptype}|{score_key}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
        deduped.append(entry)
    return deduped


# ------------------------------------------------------------------
# Load resources
# ------------------------------------------------------------------
(cfg, model, fused_index, catalog_ids,
 img_index, img_catalog_ids, item_lookup) = load_resources()

st.sidebar.caption(f"Device: **{cfg.device}**")
st.sidebar.caption(f"Catalog: **{len(catalog_ids):,}** items")
if img_index:
    st.sidebar.caption(f"Image index: **{img_index.ntotal:,}** vectors")

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
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
    search_mode = st.radio(
        "Search mode",
        ["Hybrid (image + text)", "Image only"],
        index=0,
        help="Hybrid merges visual similarity with text-aware search.",
    )
    rerank_weight = st.slider(
        "Neural vs attribute re-ranking",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="Higher = trust neural similarity more. Lower = trust attribute matching more.",
    )
    img_weight = st.slider(
        "Image similarity weight (hybrid mode)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="How much to weight pure visual similarity vs fused embedding.",
    )

# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
if uploaded is not None:
    # Show query image
    col_query, col_spacer = st.columns([1, 2])
    with col_query:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Query Image", use_container_width=True)

    # Build attribute query
    parts = []
    if brand:    parts.append(f"brand: {brand}")
    if material: parts.append(f"material: {material}")
    if color:    parts.append(f"color: {color}")
    if ptype:    parts.append(f"product_type: {ptype}")
    query_text = " | ".join(parts) if parts else "unknown product"

    if parts:
        st.markdown(f"**Attribute query:** `{query_text}`")

    if st.button("🔎 Search"):
        query_attrs = parse_attributes(query_text)
        candidate_k = min(top_k * 10, len(catalog_ids))

        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        with torch.no_grad():
            img_t = tfm(img).unsqueeze(0).to(cfg.device)
            img_emb = model.encode_image_only(img_t)
            img_np = img_emb.cpu().numpy().astype("float32")

            txt_emb = model.encode_text_only([query_attrs], cfg.device)
            fused_emb = model.fuse(img_emb, txt_emb)
            fused_np = fused_emb.cpu().numpy().astype("float32")

        # --- Search based on mode ---
        if search_mode == "Image only":
            if img_index and img_catalog_ids:
                raw_results = query_index(
                    img_np, img_index, img_catalog_ids, k=candidate_k
                )[0]
            else:
                raw_results = query_index(
                    img_np, fused_index, catalog_ids, k=candidate_k
                )[0]
        else:  # Hybrid
            fused_results = query_index(
                fused_np, fused_index, catalog_ids, k=candidate_k
            )[0]
            if img_index and img_catalog_ids:
                img_results = query_index(
                    img_np, img_index, img_catalog_ids, k=candidate_k
                )[0]
            else:
                img_results = query_index(
                    img_np, fused_index, catalog_ids, k=candidate_k
                )[0]
            merged = merge_results(fused_results, img_results,
                                   img_weight=img_weight)
            raw_results = [(score, pid) for score, pid in merged]

        # --- Re-rank, deduplicate, display ---
        reranked = attribute_rerank(
            raw_results, query_attrs, item_lookup, fusion_weight=rerank_weight
        )
        reranked = deduplicate_results(reranked, item_lookup)

        st.subheader(f"Top-{top_k} Results")
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
else:
    st.info("📷 Upload a product image to start searching. "
            "Optionally fill in attributes to refine results.")