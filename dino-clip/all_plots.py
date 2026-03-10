"""
Presentation Plots for Multimodal Retrieval Engine
===================================================
All plots use the already-trained model. NO retraining needed.

Generates:
  1. Scaling bar chart (5K vs 30K vs 148K)
  2. Retrieval examples grid (query + top-5 results)
  3. Loss comparison — InfoNCE vs PairwiseInfoNCE (single pass over val set)
  4. Fusion alpha visualization
  5. Embedding similarity distribution

Usage:
    python3 all_plots.py --max-items 5000
"""

import os, json, random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from main import (
    Config, TwoTowerModel, ABODataset, collate_fn,
    build_dataloaders, InfoNCELoss, PairwiseInfoNCE,
    parse_attributes
)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def numpy_query(query_emb, catalog_embs, catalog_ids, k=10):
    """Brute-force cosine similarity search using numpy."""
    norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.maximum(norms, 1e-8)
    norms_c = np.linalg.norm(catalog_embs, axis=1, keepdims=True)
    catalog_normed = catalog_embs / np.maximum(norms_c, 1e-8)
    sims = query_emb @ catalog_normed.T
    results = []
    for q in range(query_emb.shape[0]):
        topk_idx = np.argsort(sims[q])[::-1][:k]
        results.append([(float(sims[q, i]), catalog_ids[i]) for i in topk_idx])
    return results


# =====================================================================
# PLOT 1: Scaling Bar Chart (5K vs 30K vs 148K)
# =====================================================================

def plot_scaling_chart():
    """Hardcoded results from your three training runs."""
    print("Generating scaling chart...")
    scales = ["5K", "30K", "148K"]
    metrics = {
        "P@1":        [0.8120, 0.8693, 0.9162],
        "MAP":        [0.8233, 0.8754, 0.9177],
        "Val Cosine": [0.523,  0.640,  0.683],
    }

    x = np.arange(len(scales))
    w = 0.22
    colors = ["#2d5016", "#5a7d3a", "#DAA520"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + i * w, vals, w, label=name, color=colors[i],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(x + w)
    ax.set_xticklabels(scales, fontsize=12)
    ax.set_xlabel("Training Catalog Size", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Across Training Scales", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.15, axis="y")

    for i, (name, vals) in enumerate(metrics.items()):
        improvement = ((vals[-1] - vals[0]) / vals[0]) * 100
        ax.annotate(f"+{improvement:.1f}%",
                    xy=(x[-1] + i * w, vals[-1]),
                    xytext=(0, 15), textcoords="offset points",
                    ha="center", fontsize=8, color=colors[i], fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/scaling_chart.png", dpi=200)
    print(f"  Saved: {PLOT_DIR}/scaling_chart.png")
    plt.show()


# =====================================================================
# PLOT 2: Retrieval Examples Grid
# =====================================================================

def plot_retrieval_examples(cfg, num_queries=3, top_k=5):
    """Show query images with top-K results from the trained model."""
    print("Generating retrieval examples...")

    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    embs_path = cfg.index_save_path.replace(".bin", "_embs.npy")
    catalog_embs = np.load(embs_path)
    with open(cfg.catalog_ids_path) as f:
        catalog_ids = json.load(f)

    ds = ABODataset(cfg)
    item_lookup = {}
    for img_path, attr_text, item_id in ds.samples:
        item_lookup[item_id] = {"image_path": img_path, "attributes": attr_text}

    # Pick diverse product types
    type_examples = {}
    for img_path, attr_text, item_id in ds.samples:
        for part in attr_text.split(" | "):
            if part.strip().lower().startswith("product_type:"):
                ptype = part.split(":", 1)[1].strip()
                if ptype not in type_examples and len(type_examples) < num_queries:
                    type_examples[ptype] = (img_path, attr_text, item_id)

    if not type_examples:
        random.seed(42)
        indices = random.sample(range(len(ds.samples)), num_queries)
        queries = [ds.samples[i] for i in indices]
    else:
        queries = list(type_examples.values())[:num_queries]

    tfm = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    fig, axes = plt.subplots(num_queries, top_k + 1,
                             figsize=(4 * (top_k + 1), 4 * num_queries))
    if num_queries == 1:
        axes = axes.reshape(1, -1)

    for row, (img_path, attr_text, item_id) in enumerate(queries):
        img = Image.open(img_path).convert("RGB")
        img_t = tfm(img).unsqueeze(0).to(cfg.device)
        query_attrs = parse_attributes(attr_text)

        with torch.no_grad():
            emb = model(img_t, [query_attrs]).cpu().numpy().astype("float32")
        results = numpy_query(emb, catalog_embs, catalog_ids, k=top_k)[0]

        # Query image
        axes[row, 0].imshow(Image.open(img_path).convert("RGB"))
        short_attrs = attr_text[:60] + "..." if len(attr_text) > 60 else attr_text
        axes[row, 0].set_title(f"Query\n{short_attrs}", fontsize=8,
                                fontweight="bold", color="#2d5016")
        axes[row, 0].axis("off")
        for spine in axes[row, 0].spines.values():
            spine.set_edgecolor("#2d5016")
            spine.set_linewidth(3)
            spine.set_visible(True)

        # Extract query product type
        query_type = ""
        for part in attr_text.split(" | "):
            if "product_type" in part.lower():
                query_type = part.split(":", 1)[1].strip().upper()

        # Results
        for col, (score, pid) in enumerate(results):
            info = item_lookup.get(pid)
            if info and info["image_path"]:
                try:
                    axes[row, col + 1].imshow(
                        Image.open(info["image_path"]).convert("RGB"))
                except Exception:
                    axes[row, col + 1].text(0.5, 0.5, "No Image",
                                            ha="center", va="center")
            else:
                axes[row, col + 1].text(0.5, 0.5, "No Image",
                                        ha="center", va="center")

            result_type = ""
            if info:
                for part in info["attributes"].split(" | "):
                    if "product_type" in part.lower():
                        result_type = part.split(":", 1)[1].strip().upper()

            is_match = query_type == result_type and query_type != ""
            color = "#2d5016" if is_match else "#c44e52"
            short_result = info["attributes"][:40] + "..." if info else ""

            axes[row, col + 1].set_title(f"#{col+1} — {score:.3f}\n{short_result}",
                                          fontsize=7, color=color)
            axes[row, col + 1].axis("off")

    plt.suptitle(f"Retrieval Examples — Top-{top_k} Results",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/retrieval_examples.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/retrieval_examples.png")
    plt.show()


# =====================================================================
# PLOT 3: Loss Comparison — Single Pass Over Val Set
# =====================================================================

def plot_loss_comparison(cfg, max_items=5000):
    """
    Run a single forward pass over the val set with the trained model
    and compute per-batch InfoNCE vs PairwiseInfoNCE losses.
    No training, no gradient updates.
    """
    print("Computing loss comparison on validation set...")

    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    _, val_loader, _, _ = build_dataloaders(cfg, max_items=max_items)

    fusion_loss_fn = InfoNCELoss(cfg.temperature)
    align_loss_fn = PairwiseInfoNCE(cfg.temperature)

    batch_infonce = []
    batch_pairwise = []
    batch_cosines = []

    with torch.no_grad():
        for imgs, texts, attr_lists, _ in tqdm(val_loader, desc="Val forward pass"):
            imgs = imgs.to(cfg.device)
            img_emb = model.image_tower(imgs)
            txt_emb = model.text_tower(attr_lists, device=cfg.device)
            fused = model.fusion(img_emb, txt_emb)

            l_info = fusion_loss_fn(fused).item()
            l_pair = align_loss_fn(img_emb, txt_emb).item()
            cos = F.cosine_similarity(img_emb, txt_emb, dim=-1).mean().item()

            batch_infonce.append(l_info)
            batch_pairwise.append(l_pair)
            batch_cosines.append(cos)

    batches = range(1, len(batch_infonce) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (a) Per-batch loss comparison
    ax = axes[0]
    ax.bar(np.array(list(batches)) - 0.2, batch_infonce, 0.4,
           color="#2d5016", label="InfoNCE (fused)", alpha=0.8)
    ax.bar(np.array(list(batches)) + 0.2, batch_pairwise, 0.4,
           color="#c44e52", label="PairwiseInfoNCE (cross-modal)", alpha=0.8)
    ax.set_xlabel("Batch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Per-Batch Loss: InfoNCE vs PairwiseInfoNCE", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # (b) Distribution comparison
    ax = axes[1]
    ax.hist(batch_infonce, bins=15, alpha=0.6, color="#2d5016",
            label=f"InfoNCE\nmean={np.mean(batch_infonce):.4f}", edgecolor="white")
    ax.hist(batch_pairwise, bins=15, alpha=0.6, color="#c44e52",
            label=f"PairwiseInfoNCE\nmean={np.mean(batch_pairwise):.4f}", edgecolor="white")
    ax.set_xlabel("Loss Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Loss Distribution (Validation)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # (c) Cosine similarity per batch
    ax = axes[2]
    ax.plot(batches, batch_cosines, color="#DAA520", linewidth=2, marker="o", markersize=4)
    ax.axhline(y=np.mean(batch_cosines), color="gray", linestyle=":",
               label=f"Mean: {np.mean(batch_cosines):.4f}")
    ax.fill_between(batches, 0, batch_cosines, alpha=0.15, color="#DAA520")
    ax.set_xlabel("Batch", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Cross-Modal Alignment per Batch", fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.suptitle("Loss Analysis — Trained Model on Validation Set",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/loss_comparison.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/loss_comparison.png")
    plt.show()

    # Print summary
    print(f"\n  Summary:")
    print(f"    InfoNCE mean:     {np.mean(batch_infonce):.4f}")
    print(f"    PairwiseInfoNCE:  {np.mean(batch_pairwise):.4f}")
    print(f"    Cosine sim mean:  {np.mean(batch_cosines):.4f}")
    print(f"    Total loss:       {np.mean(batch_infonce) + np.mean(batch_pairwise):.4f}")


# =====================================================================
# PLOT 4: Fusion Alpha Visualization
# =====================================================================

def plot_alpha_snapshot(cfg):
    """Visualize the current learned fusion alpha from the trained model."""
    print("Generating alpha visualization...")

    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )

    raw_alpha = model.fusion.alpha.item()
    alpha = torch.sigmoid(model.fusion.alpha).item()
    img_pct = alpha * 100
    txt_pct = (1 - alpha) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Horizontal bar showing image vs text weight
    ax1.barh(["Fusion\nWeight"], [img_pct], color="#2d5016", height=0.5,
             label=f"Image: {img_pct:.1f}%")
    ax1.barh(["Fusion\nWeight"], [txt_pct], left=[img_pct], color="#c44e52",
             height=0.5, label=f"Text: {txt_pct:.1f}%")
    ax1.axvline(x=50, color="white", linestyle="-", linewidth=2)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Weight (%)", fontsize=12)
    ax1.set_title("Learned Fusion Balance", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11, loc="upper right")

    # Add text annotations
    ax1.text(img_pct / 2, 0, f"{img_pct:.1f}%\nImage",
             ha="center", va="center", fontsize=14, fontweight="bold", color="white")
    ax1.text(img_pct + txt_pct / 2, 0, f"{txt_pct:.1f}%\nText",
             ha="center", va="center", fontsize=14, fontweight="bold", color="white")

    # Right: Comparison with old vs new initialization
    scenarios = ["Old Init\nσ(0.5)=0.622", "Balanced\nσ(0.0)=0.500",
                 f"Learned\nσ({raw_alpha:.2f})={alpha:.3f}"]
    img_vals = [62.2, 50.0, img_pct]
    txt_vals = [37.8, 50.0, txt_pct]
    bar_colors_img = ["#8B0000", "#808080", "#2d5016"]
    bar_colors_txt = ["#c44e52", "#a0a0a0", "#c44e52"]

    x = np.arange(len(scenarios))
    ax2.bar(x, img_vals, 0.6, color=bar_colors_img, alpha=0.8, label="Image %")
    ax2.bar(x, txt_vals, 0.6, bottom=img_vals, color=bar_colors_txt,
            alpha=0.5, label="Text %")
    ax2.axhline(y=50, color="white", linestyle="-", linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=10)
    ax2.set_ylabel("Weight (%)", fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.set_title("Initialization Comparison", fontsize=13, fontweight="bold")

    for i, (iv, tv) in enumerate(zip(img_vals, txt_vals)):
        ax2.text(i, iv / 2, f"{iv:.1f}%", ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white")
        ax2.text(i, iv + tv / 2, f"{tv:.1f}%", ha="center", va="center",
                 fontsize=10, fontweight="bold", color="white")

    plt.suptitle("Fusion Head: Learned Image vs Text Weighting",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/alpha_snapshot.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/alpha_snapshot.png")
    plt.show()

    print(f"\n  Raw alpha: {raw_alpha:.4f}")
    print(f"  Sigmoid(alpha): {alpha:.4f}")
    print(f"  Image weight: {img_pct:.1f}%")
    print(f"  Text weight: {txt_pct:.1f}%")


# =====================================================================
# PLOT 5: Embedding Similarity Distribution
# =====================================================================

def plot_similarity_distribution(cfg, max_items=2000):
    """
    Show distribution of cosine similarities:
    - Matching pairs (same product image-text)
    - Random pairs (different products)
    """
    print("Computing similarity distributions...")

    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    ds = ABODataset(cfg, max_items=max_items)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    all_img_embs, all_txt_embs = [], []
    with torch.no_grad():
        for imgs, texts, attr_lists, _ in tqdm(loader, desc="Encoding"):
            imgs = imgs.to(cfg.device)
            all_img_embs.append(model.encode_image_only(imgs).cpu())
            all_txt_embs.append(model.encode_text_only(attr_lists, device=cfg.device).cpu())

    img_embs = torch.cat(all_img_embs)
    txt_embs = torch.cat(all_txt_embs)
    n = len(img_embs)

    # Matching pairs: image_i with text_i (same product)
    matching_sims = F.cosine_similarity(img_embs, txt_embs, dim=-1).numpy()

    # Random pairs: image_i with text_j where j != i
    random.seed(42)
    perm = list(range(n))
    random.shuffle(perm)
    # Ensure no self-matches
    for i in range(n):
        if perm[i] == i:
            perm[i] = (i + 1) % n
    shuffled_txt = txt_embs[perm]
    random_sims = F.cosine_similarity(img_embs, shuffled_txt, dim=-1).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Overlaid histograms
    ax1.hist(matching_sims, bins=50, alpha=0.6, color="#2d5016",
             label=f"Matching pairs\nmean={matching_sims.mean():.3f}", density=True)
    ax1.hist(random_sims, bins=50, alpha=0.6, color="#c44e52",
             label=f"Random pairs\nmean={random_sims.mean():.3f}", density=True)
    ax1.set_xlabel("Cosine Similarity", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Image-Text Similarity Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Separation metric
    separation = matching_sims.mean() - random_sims.mean()
    ax1.annotate(f"Separation: {separation:.3f}",
                 xy=(0.5, 0.9), xycoords="axes fraction",
                 fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#DAA520", alpha=0.3))

    # (b) Box plot comparison
    ax2.boxplot([matching_sims, random_sims],
                labels=["Matching\n(same product)", "Random\n(different products)"],
                patch_artist=True,
                boxprops=[dict(facecolor="#2d5016", alpha=0.5),
                          dict(facecolor="#c44e52", alpha=0.5)],
                medianprops=dict(color="black", linewidth=2))
    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_title("Matching vs Random Pair Similarity", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Cross-Modal Embedding Quality",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/similarity_distribution.png", dpi=200, bbox_inches="tight")
    print(f"  Saved: {PLOT_DIR}/similarity_distribution.png")
    plt.show()

    print(f"\n  Matching pairs mean: {matching_sims.mean():.4f}")
    print(f"  Random pairs mean:   {random_sims.mean():.4f}")
    print(f"  Separation:          {separation:.4f}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate all presentation plots")
    parser.add_argument("--max-items", type=int, default=5000,
                        help="Items to load for loss/similarity plots")
    args = parser.parse_args()

    cfg = Config()
    print(f"Device: {cfg.device}\n")

    # 1. Scaling chart (instant, hardcoded)
    print("=" * 60)
    plot_scaling_chart()

    # 2. Retrieval examples (needs model + index)
    print("\n" + "=" * 60)
    plot_retrieval_examples(cfg, num_queries=3, top_k=5)

    # 3. Loss comparison (single val pass, no training)
    print("\n" + "=" * 60)
    plot_loss_comparison(cfg, max_items=args.max_items)

    # 4. Alpha snapshot (instant, reads model weights)
    print("\n" + "=" * 60)
    plot_alpha_snapshot(cfg)

    # 5. Similarity distribution (single forward pass)
    print("\n" + "=" * 60)
    plot_similarity_distribution(cfg, max_items=args.max_items)

    print(f"\n{'=' * 60}")
    print(f"All plots saved to {PLOT_DIR}/")
    print(f"{'=' * 60}")