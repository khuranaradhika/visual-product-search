"""
t-SNE Visualization of Multimodal Embeddings
=============================================
Run after training + indexing:
    python tsne_plot.py

Generates 3 plots:
  1. Fused embeddings colored by product type
  2. Image vs Text embeddings (modality gap visualization)
  3. Fused embeddings colored by a specific attribute (e.g., color)
"""

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm
from torch.utils.data import DataLoader

from main import (
    Config, TwoTowerModel, ABODataset, collate_fn, parse_attributes
)


def load_model_and_data(cfg, max_items=5000):
    """Load trained model and a subset of data for visualization."""
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()
    ds = ABODataset(cfg, max_items=max_items)
    return model, ds


def extract_embeddings(model, dataset, cfg, max_samples=2000):
    """Extract fused, image-only, and text-only embeddings."""
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    fused_embs, img_embs, txt_embs = [], [], []
    all_texts, all_ids = [], []

    with torch.no_grad():
        for imgs, texts, attr_lists, ids in tqdm(loader, desc="Extracting embeddings"):
            if len(all_ids) >= max_samples:
                break
            imgs = imgs.to(cfg.device)
            img_e = model.encode_image_only(imgs).cpu().numpy()
            txt_e = model.encode_text_only(attr_lists, device=cfg.device).cpu().numpy()
            fused_e = model.fusion(
                model.encode_image_only(imgs),
                model.encode_text_only(attr_lists, device=cfg.device)
            ).cpu().numpy()

            img_embs.append(img_e)
            txt_embs.append(txt_e)
            fused_embs.append(fused_e)
            all_texts.extend(texts)
            all_ids.extend(ids)

    fused_embs = np.vstack(fused_embs)[:max_samples]
    img_embs = np.vstack(img_embs)[:max_samples]
    txt_embs = np.vstack(txt_embs)[:max_samples]
    all_texts = all_texts[:max_samples]
    all_ids = all_ids[:max_samples]

    return fused_embs, img_embs, txt_embs, all_texts, all_ids


def extract_attribute(attr_text, key):
    """Pull a specific attribute value from the attribute string."""
    for part in attr_text.split(" | "):
        if part.strip().lower().startswith(f"{key}:"):
            return part.split(":", 1)[1].strip()
    return "Unknown"


def plot_tsne_by_product_type(fused_embs, all_texts, top_n=10, save_path="tsne_product_type.png"):
    """Plot 1: Fused embeddings colored by product type."""
    print("Computing t-SNE for product types...")
    types = [extract_attribute(t, "product_type") for t in all_texts]

    # Keep only the top N most common types for clarity
    type_counts = Counter(types)
    top_types = [t for t, _ in type_counts.most_common(top_n)]

    # Filter to top types only
    mask = [i for i, t in enumerate(types) if t in top_types]
    filtered_embs = fused_embs[mask]
    filtered_types = [types[i] for i in mask]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(filtered_embs)

    # Color map
    unique_types = sorted(set(filtered_types))
    cmap = plt.cm.get_cmap("tab10", len(unique_types))
    type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for t in unique_types:
        idx = [i for i, ft in enumerate(filtered_types) if ft == t]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=[type_to_color[t]], label=t, s=15, alpha=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, markerscale=2)
    ax.set_title("t-SNE of Fused Embeddings by Product Type", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_tsne_modality_gap(img_embs, txt_embs, n_samples=1000, save_path="tsne_modality_gap.png"):
    """Plot 2: Image vs Text embeddings showing modality gap."""
    print("Computing t-SNE for modality gap...")

    n = min(n_samples, len(img_embs))
    combined = np.vstack([img_embs[:n], txt_embs[:n]])
    labels = ["Image"] * n + ["Text"] * n

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 8))

    img_coords = coords[:n]
    txt_coords = coords[n:]

    ax.scatter(img_coords[:, 0], img_coords[:, 1],
               c='#2d5016', label='Image Embeddings', s=10, alpha=0.5)
    ax.scatter(txt_coords[:, 0], txt_coords[:, 1],
               c='#c44e52', label='Text Embeddings', s=10, alpha=0.5)

    # Draw lines connecting matching pairs (same product)
    for i in range(0, n, max(1, n // 50)):  # draw ~50 connecting lines
        ax.plot([img_coords[i, 0], txt_coords[i, 0]],
                [img_coords[i, 1], txt_coords[i, 1]],
                color='gray', alpha=0.2, linewidth=0.5)

    ax.legend(fontsize=12, markerscale=3)
    ax.set_title("t-SNE: Image vs Text Embeddings (Modality Gap)", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_tsne_by_color(fused_embs, all_texts, save_path="tsne_color.png"):
    """Plot 3: Fused embeddings colored by product color attribute."""
    print("Computing t-SNE for color attributes...")
    colors = [extract_attribute(t, "color") for t in all_texts]

    # Map common colors to actual plot colors
    color_map = {
        "Black": "#1a1a1a", "White": "#d4d4d4", "Brown": "#8B4513",
        "Blue": "#4169E1", "Red": "#DC143C", "Green": "#228B22",
        "Grey": "#808080", "Gray": "#808080", "Silver": "#C0C0C0",
        "Navy": "#000080", "Beige": "#F5F5DC", "Pink": "#FF69B4",
        "Yellow": "#FFD700", "Gold": "#DAA520", "Natural": "#D2B48C",
    }

    # Keep only items with recognizable colors
    mask = [i for i, c in enumerate(colors) if c in color_map]
    if len(mask) < 50:
        print("Not enough color-labeled items for meaningful plot. Skipping.")
        return

    filtered_embs = fused_embs[mask]
    filtered_colors = [colors[i] for i in mask]
    plot_colors = [color_map[c] for c in filtered_colors]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(filtered_embs)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each color group
    for color_name in sorted(set(filtered_colors)):
        idx = [i for i, c in enumerate(filtered_colors) if c == color_name]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   c=color_map[color_name], label=color_name, s=20, alpha=0.7,
                   edgecolors='black', linewidths=0.3)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, markerscale=2)
    ax.set_title("t-SNE of Fused Embeddings by Product Color", fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    cfg = Config()
    print(f"Device: {cfg.device}")

    # Load model and data (use 5K subset for fast visualization)
    model, ds = load_model_and_data(cfg, max_items=5000)

    # Extract all embedding types
    fused_embs, img_embs, txt_embs, all_texts, all_ids = extract_embeddings(
        model, ds, cfg, max_samples=2000
    )

    print(f"\nEmbeddings extracted: {fused_embs.shape[0]} samples")

    # Generate all three plots
    plot_tsne_by_product_type(fused_embs, all_texts, top_n=10)
    plot_tsne_modality_gap(img_embs, txt_embs, n_samples=1000)
    plot_tsne_by_color(fused_embs, all_texts)

    print("\nAll plots saved!")