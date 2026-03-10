"""
Interactive Embedding Explorer for Multimodal Retrieval Engine
==============================================================
Uses embedding-explorer to create an interactive dashboard
for exploring your product embeddings.

Install:
    pip install embedding-explorer

Usage:
    python3 embedding_explorer_app.py --max-items 5000

Opens a browser with interactive clustering and network views.
"""

import os, json, argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from main import (
    Config, TwoTowerModel, ABODataset, collate_fn, parse_attributes
)


def extract_embeddings_and_metadata(cfg, max_items=5000):
    """Extract embeddings and metadata from trained model."""
    print("Loading model...")
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    print(f"Loading dataset (max {max_items} items)...")
    ds = ABODataset(cfg, max_items=max_items)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    fused_embs, img_embs, txt_embs = [], [], []
    all_texts, all_ids = [], []

    with torch.no_grad():
        for imgs, texts, attr_lists, ids in tqdm(loader, desc="Extracting embeddings"):
            imgs = imgs.to(cfg.device)
            img_e = model.encode_image_only(imgs)
            txt_e = model.encode_text_only(attr_lists, device=cfg.device)
            fused_e = model.fusion(img_e, txt_e)

            fused_embs.append(fused_e.cpu().numpy())
            img_embs.append(img_e.cpu().numpy())
            txt_embs.append(txt_e.cpu().numpy())
            all_texts.extend(texts)
            all_ids.extend(ids)

    fused_embs = np.vstack(fused_embs)
    img_embs = np.vstack(img_embs)
    txt_embs = np.vstack(txt_embs)

    print(f"Extracted {len(all_ids)} embeddings (dim={fused_embs.shape[1]})")
    return fused_embs, img_embs, txt_embs, all_texts, all_ids


def extract_attribute(attr_text, key):
    """Pull a specific attribute from the attribute string."""
    for part in attr_text.split(" | "):
        if part.strip().lower().startswith(f"{key}:"):
            return part.split(":", 1)[1].strip()
    return "Unknown"


def build_metadata(all_texts, all_ids):
    """Build a pandas DataFrame of metadata for the explorer."""
    import pandas as pd

    records = []
    for text, item_id in zip(all_texts, all_ids):
        records.append({
            "item_id": item_id,
            "product_type": extract_attribute(text, "product_type"),
            "color": extract_attribute(text, "color"),
            "material": extract_attribute(text, "material"),
            "brand": extract_attribute(text, "brand"),
            "full_attributes": text[:100],
        })

    return pd.DataFrame(records)


def run_explorer(cfg, max_items=5000):
    """Launch the embedding-explorer dashboard."""
    try:
        from embedding_explorer import show_clustering
        from embedding_explorer.cards import ClusteringCard, NetworkCard
        from embedding_explorer import show_dashboard
    except ImportError:
        print("embedding-explorer not installed. Install with:")
        print("  pip install embedding-explorer")
        print("\nFalling back to standalone Plotly visualization...")
        run_plotly_fallback(cfg, max_items)
        return

    fused_embs, img_embs, txt_embs, all_texts, all_ids = \
        extract_embeddings_and_metadata(cfg, max_items)

    metadata = build_metadata(all_texts, all_ids)

    # Create corpus labels for the explorer
    corpus = [
        f"{row['product_type']} | {row['color']} | {row['brand']}"
        for _, row in metadata.iterrows()
    ]

    print("\nLaunching embedding-explorer dashboard...")
    print("Open your browser to the URL shown below.\n")

    # Build cards for dashboard
    cards = [
        ClusteringCard(
            name="Fused Embeddings (Image + Text)",
            corpus=corpus,
            embeddings=fused_embs,
            metadata=metadata,
            hover_name="full_attributes",
            hover_data=["product_type", "color", "material", "brand"],
        ),
        ClusteringCard(
            name="Image-Only Embeddings (DINOv3)",
            corpus=corpus,
            embeddings=img_embs,
            metadata=metadata,
            hover_name="full_attributes",
            hover_data=["product_type", "color"],
        ),
        ClusteringCard(
            name="Text-Only Embeddings (CLIP)",
            corpus=corpus,
            embeddings=txt_embs,
            metadata=metadata,
            hover_name="full_attributes",
            hover_data=["product_type", "color"],
        ),
        NetworkCard(
            name="Semantic Network (Fused)",
            corpus=corpus,
            embeddings=fused_embs,
        ),
    ]

    show_dashboard(cards, port=8060)
    print("Dashboard running at http://localhost:8060")
    input("Press Enter to stop the server...")


def run_plotly_fallback(cfg, max_items=5000):
    """
    Fallback: Interactive Plotly visualization if embedding-explorer
    is not installed. Saves an HTML file you can open in browser.
    """
    try:
        import plotly.express as px
    except ImportError:
        print("Plotly not installed either. Install with:")
        print("  pip install plotly")
        return

    from sklearn.manifold import TSNE
    import pandas as pd

    fused_embs, img_embs, txt_embs, all_texts, all_ids = \
        extract_embeddings_and_metadata(cfg, max_items)

    metadata = build_metadata(all_texts, all_ids)

    # Limit for t-SNE speed
    n = min(3000, len(fused_embs))
    embs = fused_embs[:n]
    meta = metadata.iloc[:n].copy()

    print(f"Running t-SNE on {n} embeddings...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embs)

    meta["tsne_x"] = coords[:, 0]
    meta["tsne_y"] = coords[:, 1]

    # Interactive scatter plot colored by product type
    fig = px.scatter(
        meta, x="tsne_x", y="tsne_y",
        color="product_type",
        hover_data=["item_id", "color", "material", "brand", "full_attributes"],
        title="Interactive Embedding Explorer — Fused Embeddings by Product Type",
        width=1200, height=800,
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        legend=dict(font=dict(size=9)),
        hoverlabel=dict(font_size=10),
    )

    html_path = "plots/embedding_explorer.html"
    os.makedirs("plots", exist_ok=True)
    fig.write_html(html_path)
    print(f"\nSaved interactive plot: {html_path}")
    print("Open this file in your browser to explore!")

    # Also make a color version
    fig2 = px.scatter(
        meta, x="tsne_x", y="tsne_y",
        color="color",
        hover_data=["item_id", "product_type", "material", "brand", "full_attributes"],
        title="Interactive Embedding Explorer — Fused Embeddings by Color",
        width=1200, height=800,
        opacity=0.7,
    )
    fig2.update_traces(marker=dict(size=5))
    fig2.update_layout(
        legend=dict(font=dict(size=9)),
        hoverlabel=dict(font_size=10),
    )

    html_path2 = "plots/embedding_explorer_color.html"
    fig2.write_html(html_path2)
    print(f"Saved interactive plot: {html_path2}")

    # Brand version
    fig3 = px.scatter(
        meta, x="tsne_x", y="tsne_y",
        color="brand",
        hover_data=["item_id", "product_type", "color", "material", "full_attributes"],
        title="Interactive Embedding Explorer — Fused Embeddings by Brand",
        width=1200, height=800,
        opacity=0.7,
    )
    fig3.update_traces(marker=dict(size=5))
    fig3.update_layout(
        legend=dict(font=dict(size=9)),
        hoverlabel=dict(font_size=10),
    )

    html_path3 = "plots/embedding_explorer_brand.html"
    fig3.write_html(html_path3)
    print(f"Saved interactive plot: {html_path3}")

    print("\nAll interactive plots saved to plots/ directory!")
    print("Open the HTML files in your browser — you can hover, zoom, and click on points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--fallback", action="store_true",
                        help="Skip embedding-explorer, use Plotly directly")
    args = parser.parse_args()

    cfg = Config()
    print(f"Device: {cfg.device}\n")

    if args.fallback:
        run_plotly_fallback(cfg, max_items=args.max_items)
    else:
        run_explorer(cfg, max_items=args.max_items)