"""
NDCG Evaluation with Attribute-Weighted Graded Relevance
========================================================
Uses already-trained model and index. No retraining needed.

Relevance scoring:
    +1 for matching product_type
    +1 for matching color
    +1 for matching material
    +1 for matching brand
    Max relevance = 4 per item

Usage:
    python3 ndcg_eval.py --max-items 5000

Output:
    - NDCG@1, NDCG@5, NDCG@10 scores
    - Comparison chart: binary P@K vs graded NDCG@K
    - Per-attribute contribution analysis
"""

import os, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import (
    Config, TwoTowerModel, ABODataset, collate_fn,
    build_dataloaders, parse_attributes
)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def extract_attr_dict(attr_text):
    """Parse 'brand: X | color: Y | ...' into {'brand': 'x', 'color': 'y', ...}"""
    d = {}
    for part in attr_text.split(" | "):
        if ":" in part:
            key, val = part.split(":", 1)
            d[key.strip().lower()] = val.strip().lower()
    return d


def graded_relevance(query_attrs, candidate_attrs):
    """
    Compute graded relevance score between query and candidate.
    +1 for each matching attribute (product_type, color, material, brand).
    Returns score 0-4.
    """
    score = 0
    keys_to_check = ["product_type", "color", "material", "brand"]

    for key in keys_to_check:
        q_val = query_attrs.get(key, "")
        c_val = candidate_attrs.get(key, "")
        if q_val and c_val:
            # Exact match (case insensitive, already lowered)
            if q_val == c_val:
                score += 1
            # Partial / substring match
            elif q_val in c_val or c_val in q_val:
                score += 0.5
    return score


def dcg_at_k(relevances, k):
    """Discounted Cumulative Gain at K."""
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0
    # DCG = sum of rel_i / log2(i + 1) for i = 1..k
    discounts = np.log2(np.arange(1, len(relevances) + 1) + 1)
    return np.sum(relevances / discounts)


def ndcg_at_k(relevances, k):
    """Normalized DCG at K."""
    actual_dcg = dcg_at_k(relevances, k)
    # Ideal: sort relevances descending
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def numpy_query(query_emb, catalog_embs, catalog_ids, k=10):
    """Brute-force cosine similarity search."""
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


def evaluate_ndcg(cfg, max_items=5000):
    """Run full NDCG evaluation on the test set."""

    # Load model
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    # Load catalog embeddings
    embs_path = cfg.index_save_path.replace(".bin", "_embs.npy")
    catalog_embs = np.load(embs_path)
    with open(cfg.catalog_ids_path) as f:
        catalog_ids = json.load(f)

    # Build attribute lookup for catalog
    ds = ABODataset(cfg, max_items=max_items)
    catalog_attrs = {}
    for _, attr_text, item_id in ds.samples:
        catalog_attrs[item_id] = extract_attr_dict(attr_text)

    # Build test set
    _, _, test_loader, _ = build_dataloaders(cfg, max_items=max_items)

    k_values = [1, 5, 10]
    max_k = max(k_values)

    # Storage
    ndcg_scores = {f"NDCG@{k}": [] for k in k_values}
    precision_scores = {f"P@{k}": [] for k in k_values}
    per_attr_matches = {"product_type": 0, "color": 0, "material": 0, "brand": 0}
    per_attr_total = {"product_type": 0, "color": 0, "material": 0, "brand": 0}
    all_relevances = []

    with torch.no_grad():
        for imgs, texts, attr_lists, gt_ids in tqdm(test_loader, desc="NDCG Eval"):
            imgs = imgs.to(cfg.device)
            embs = model(imgs, attr_lists).cpu().numpy().astype("float32")
            batch_results = numpy_query(embs, catalog_embs, catalog_ids, k=max_k)

            for i, (gt_id, query_text) in enumerate(zip(gt_ids, texts)):
                query_dict = extract_attr_dict(query_text)
                retrieved_ids = [r[1] for r in batch_results[i]]

                # Compute graded relevance for each retrieved item
                relevances = []
                for rid in retrieved_ids:
                    cand_dict = catalog_attrs.get(rid, {})
                    rel = graded_relevance(query_dict, cand_dict)
                    relevances.append(rel)

                all_relevances.append(relevances)

                # NDCG at each K
                for k in k_values:
                    ndcg_scores[f"NDCG@{k}"].append(ndcg_at_k(relevances, k))

                # Binary P@K for comparison (product_type match only)
                query_type = query_dict.get("product_type", "")
                for k in k_values:
                    hits = sum(
                        1 for rid in retrieved_ids[:k]
                        if catalog_attrs.get(rid, {}).get("product_type", "") == query_type
                    )
                    precision_scores[f"P@{k}"].append(hits / k)

                # Per-attribute match tracking (top-1 result)
                if retrieved_ids:
                    top1_dict = catalog_attrs.get(retrieved_ids[0], {})
                    for attr in per_attr_matches:
                        q_val = query_dict.get(attr, "")
                        c_val = top1_dict.get(attr, "")
                        if q_val:
                            per_attr_total[attr] += 1
                            if q_val == c_val:
                                per_attr_matches[attr] += 1

    # Compute means
    ndcg_means = {k: np.mean(v) for k, v in ndcg_scores.items()}
    precision_means = {k: np.mean(v) for k, v in precision_scores.items()}

    # Print results
    print("\n" + "=" * 60)
    print("NDCG Results (Graded Relevance: product_type + color + material + brand)")
    print("=" * 60)
    for k in k_values:
        print(f"  NDCG@{k:2d}: {ndcg_means[f'NDCG@{k}']:.4f}    "
              f"P@{k:2d}: {precision_means[f'P@{k}']:.4f}")

    print(f"\n  Per-Attribute Match Rate (Top-1 Result):")
    for attr in per_attr_matches:
        total = per_attr_total[attr]
        if total > 0:
            rate = per_attr_matches[attr] / total
            print(f"    {attr:15s}: {rate:.4f} ({per_attr_matches[attr]}/{total})")
        else:
            print(f"    {attr:15s}: N/A (no queries with this attribute)")

    return ndcg_means, precision_means, per_attr_matches, per_attr_total, all_relevances


def plot_ndcg_results(ndcg_means, precision_means, per_attr_matches, per_attr_total):
    """Generate comparison plots."""

    k_values = [1, 5, 10]

    # ---- Plot 1: NDCG vs P@K comparison ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(k_values))
    w = 0.3

    ndcg_vals = [ndcg_means[f"NDCG@{k}"] for k in k_values]
    pk_vals = [precision_means[f"P@{k}"] for k in k_values]

    bars1 = ax1.bar(x - w/2, pk_vals, w, color="#2d5016", label="P@K (binary: category only)")
    bars2 = ax1.bar(x + w/2, ndcg_vals, w, color="#DAA520", label="NDCG@K (graded: 4 attributes)")

    for bar, val in zip(bars1, pk_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, ndcg_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in k_values], fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Binary P@K vs Graded NDCG@K", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.15, axis="y")

    # Gap annotation
    for i, k in enumerate(k_values):
        gap = pk_vals[i] - ndcg_vals[i]
        if gap > 0:
            ax1.annotate(f"gap: {gap:.3f}",
                         xy=(x[i], min(pk_vals[i], ndcg_vals[i]) - 0.03),
                         ha="center", fontsize=8, color="#c44e52")

    # ---- Plot 2: Per-attribute match rate ----
    attrs = list(per_attr_matches.keys())
    rates = []
    for attr in attrs:
        total = per_attr_total[attr]
        rates.append(per_attr_matches[attr] / total if total > 0 else 0)

    colors = ["#2d5016", "#c44e52", "#DAA520", "#4169E1"]
    bars = ax2.bar(attrs, rates, color=colors, alpha=0.8, edgecolor="white")

    for bar, rate, attr in zip(bars, rates, attrs):
        total = per_attr_total[attr]
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{rate:.1%}\n({per_attr_matches[attr]}/{total})",
                 ha="center", fontsize=9, fontweight="bold")

    ax2.set_ylabel("Top-1 Match Rate", fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Per-Attribute Match Rate (Top-1 Result)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.15, axis="y")

    plt.suptitle("Evaluation: Binary vs Graded Relevance",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/ndcg_evaluation.png", dpi=200, bbox_inches="tight")
    print(f"\nSaved: {PLOT_DIR}/ndcg_evaluation.png")
    plt.show()


def plot_relevance_distribution(all_relevances):
    """Show distribution of relevance scores across retrieved items."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Top-1 relevance distribution
    top1_rels = [r[0] if r else 0 for r in all_relevances]
    ax1.hist(top1_rels, bins=np.arange(-0.25, 4.75, 0.5), color="#2d5016",
             alpha=0.8, edgecolor="white", rwidth=0.8)
    ax1.set_xlabel("Relevance Score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title(f"Top-1 Relevance Distribution\nmean={np.mean(top1_rels):.2f}",
                  fontsize=13, fontweight="bold")
    ax1.set_xticks([0, 1, 2, 3, 4])
    ax1.set_xticklabels(["0\n(no match)", "1\n(1 attr)", "2\n(2 attrs)",
                          "3\n(3 attrs)", "4\n(all 4)"])
    ax1.grid(True, alpha=0.2, axis="y")

    # Average relevance by position
    max_k = 10
    avg_by_pos = []
    for pos in range(max_k):
        rels_at_pos = [r[pos] for r in all_relevances if len(r) > pos]
        avg_by_pos.append(np.mean(rels_at_pos) if rels_at_pos else 0)

    ax2.bar(range(1, max_k + 1), avg_by_pos, color="#DAA520", alpha=0.8,
            edgecolor="white")
    ax2.set_xlabel("Result Position", fontsize=12)
    ax2.set_ylabel("Average Relevance", fontsize=12)
    ax2.set_title("Average Relevance by Rank Position", fontsize=13, fontweight="bold")
    ax2.set_xticks(range(1, max_k + 1))
    ax2.grid(True, alpha=0.2, axis="y")

    plt.suptitle("Graded Relevance Analysis",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/relevance_distribution.png", dpi=200, bbox_inches="tight")
    print(f"Saved: {PLOT_DIR}/relevance_distribution.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=5000)
    args = parser.parse_args()

    cfg = Config()
    print(f"Device: {cfg.device}\n")

    ndcg_means, precision_means, per_attr_matches, per_attr_total, all_relevances = \
        evaluate_ndcg(cfg, max_items=args.max_items)

    plot_ndcg_results(ndcg_means, precision_means, per_attr_matches, per_attr_total)
    plot_relevance_distribution(all_relevances)

    print(f"\nAll plots saved to {PLOT_DIR}/")