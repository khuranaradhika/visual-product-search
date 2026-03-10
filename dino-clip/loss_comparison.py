"""
InfoNCE vs PairwiseInfoNCE Loss Tracking & Visualization
=========================================================
Run after training to generate comparison plots.

Usage:
    python loss_comparison.py --max-items 5000 --epochs 10

This trains the model while logging each loss component separately,
then generates plots showing how each loss evolves over training.
"""

import os, random, json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from main import (
    Config, TwoTowerModel, ABODataset, collate_fn,
    build_dataloaders, InfoNCELoss, PairwiseInfoNCE,
    _build_param_groups, parse_attributes
)


def track_losses_per_epoch(cfg, max_items=None, epochs=None):
    """
    Train the model while tracking InfoNCE and PairwiseInfoNCE
    losses separately for visualization.
    """
    if epochs:
        cfg.epochs = epochs

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader, val_loader, test_loader, full_ds = build_dataloaders(
        cfg, max_items=max_items
    )
    print(f"[Track] {len(train_loader.dataset)} train, "
          f"{len(val_loader.dataset)} val")

    model = TwoTowerModel(cfg).to(cfg.device)
    fusion_loss_fn = InfoNCELoss(cfg.temperature)
    align_loss_fn = PairwiseInfoNCE(cfg.temperature)

    param_groups = _build_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    # Storage for per-epoch metrics
    history = {
        "epoch": [],
        "train_infonce": [],
        "train_pairwise": [],
        "train_total": [],
        "val_infonce": [],
        "val_pairwise": [],
        "val_total": [],
        "val_cosine": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        model.train()
        epoch_infonce, epoch_pairwise, epoch_total = [], [], []

        for imgs, texts, attr_lists, _ in tqdm(train_loader,
                                                 desc=f"Epoch {epoch} Train",
                                                 leave=False):
            imgs = imgs.to(cfg.device)
            img_emb = model.image_tower(imgs)
            txt_emb = model.text_tower(attr_lists, device=cfg.device)
            fused = model.fusion(img_emb, txt_emb)

            l_infonce = fusion_loss_fn(fused)
            l_pairwise = align_loss_fn(img_emb, txt_emb)
            loss = l_infonce + 1.0 * l_pairwise

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_infonce.append(l_infonce.item())
            epoch_pairwise.append(l_pairwise.item())
            epoch_total.append(loss.item())

        # --- Validate ---
        model.eval()
        val_infonce, val_pairwise, val_total = [], [], []
        cosines = []

        with torch.no_grad():
            for imgs, texts, attr_lists, _ in val_loader:
                imgs = imgs.to(cfg.device)
                img_emb = model.image_tower(imgs)
                txt_emb = model.text_tower(attr_lists, device=cfg.device)
                fused = model.fusion(img_emb, txt_emb)

                l_infonce = fusion_loss_fn(fused)
                l_pairwise = align_loss_fn(img_emb, txt_emb)

                val_infonce.append(l_infonce.item())
                val_pairwise.append(l_pairwise.item())
                val_total.append((l_infonce + 1.0 * l_pairwise).item())

                cos = F.cosine_similarity(img_emb, txt_emb, dim=-1)
                cosines.append(cos.cpu())

        scheduler.step()

        # Log
        history["epoch"].append(epoch)
        history["train_infonce"].append(np.mean(epoch_infonce))
        history["train_pairwise"].append(np.mean(epoch_pairwise))
        history["train_total"].append(np.mean(epoch_total))
        history["val_infonce"].append(np.mean(val_infonce))
        history["val_pairwise"].append(np.mean(val_pairwise))
        history["val_total"].append(np.mean(val_total))
        history["val_cosine"].append(torch.cat(cosines).mean().item())

        print(f"Epoch {epoch:3d} | "
              f"InfoNCE: {history['train_infonce'][-1]:.4f} / {history['val_infonce'][-1]:.4f} | "
              f"Pairwise: {history['train_pairwise'][-1]:.4f} / {history['val_pairwise'][-1]:.4f} | "
              f"Total: {history['train_total'][-1]:.4f} / {history['val_total'][-1]:.4f} | "
              f"Cosine: {history['val_cosine'][-1]:.4f}")

    return history


def plot_loss_comparison(history, save_dir="plots"):
    """Generate all comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = history["epoch"]

    # ---- Plot 1: InfoNCE vs PairwiseInfoNCE (Training) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_infonce"],
            color="#2d5016", linewidth=2.5, label="InfoNCE (fused embeddings)",
            marker="o", markersize=5)
    ax.plot(epochs, history["train_pairwise"],
            color="#c44e52", linewidth=2.5, label="PairwiseInfoNCE (cross-modal)",
            marker="s", markersize=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss: InfoNCE vs PairwiseInfoNCE", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_loss_comparison.png", dpi=200)
    print(f"Saved: {save_dir}/train_loss_comparison.png")
    plt.show()

    # ---- Plot 2: InfoNCE vs PairwiseInfoNCE (Validation) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["val_infonce"],
            color="#2d5016", linewidth=2.5, label="InfoNCE (fused embeddings)",
            marker="o", markersize=5)
    ax.plot(epochs, history["val_pairwise"],
            color="#c44e52", linewidth=2.5, label="PairwiseInfoNCE (cross-modal)",
            marker="s", markersize=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Validation Loss: InfoNCE vs PairwiseInfoNCE", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/val_loss_comparison.png", dpi=200)
    print(f"Saved: {save_dir}/val_loss_comparison.png")
    plt.show()

    # ---- Plot 3: Train vs Val for each loss (overfitting check) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs, history["train_infonce"],
             color="#2d5016", linewidth=2, label="Train", linestyle="-")
    ax1.plot(epochs, history["val_infonce"],
             color="#2d5016", linewidth=2, label="Val", linestyle="--")
    ax1.set_title("InfoNCE: Train vs Val", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    ax2.plot(epochs, history["train_pairwise"],
             color="#c44e52", linewidth=2, label="Train", linestyle="-")
    ax2.plot(epochs, history["val_pairwise"],
             color="#c44e52", linewidth=2, label="Val", linestyle="--")
    ax2.set_title("PairwiseInfoNCE: Train vs Val", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_trainval_split.png", dpi=200)
    print(f"Saved: {save_dir}/loss_trainval_split.png")
    plt.show()

    # ---- Plot 4: Total loss + Cosine similarity (dual axis) ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(epochs, history["val_total"],
             color="#2d5016", linewidth=2.5, label="Val Total Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss", fontsize=12, color="#2d5016")
    ax1.tick_params(axis="y", labelcolor="#2d5016")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_cosine"],
             color="#DAA520", linewidth=2.5, label="Val Cosine Similarity",
             linestyle="--")
    ax2.set_ylabel("Cosine Similarity", fontsize=12, color="#DAA520")
    ax2.tick_params(axis="y", labelcolor="#DAA520")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=11)

    plt.title("Validation: Total Loss vs Cross-Modal Alignment", fontsize=14)
    ax1.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_vs_cosine.png", dpi=200)
    print(f"Saved: {save_dir}/loss_vs_cosine.png")
    plt.show()

    # ---- Plot 5: Stacked area — loss composition ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(epochs, 0, history["train_infonce"],
                    alpha=0.4, color="#2d5016", label="InfoNCE")
    ax.fill_between(epochs, history["train_infonce"],
                    [a + b for a, b in zip(history["train_infonce"],
                                            history["train_pairwise"])],
                    alpha=0.4, color="#c44e52", label="PairwiseInfoNCE")
    ax.plot(epochs, history["train_total"],
            color="black", linewidth=2, label="Total Loss", linestyle="-")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Composition: InfoNCE + PairwiseInfoNCE", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/loss_composition.png", dpi=200)
    print(f"Saved: {save_dir}/loss_composition.png")
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    cfg = Config()
    print(f"Device: {cfg.device}")

    history = track_losses_per_epoch(cfg, max_items=args.max_items, epochs=args.epochs)

    # Save history for later use
    os.makedirs("plots", exist_ok=True)
    with open("plots/loss_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved: plots/loss_history.json")

    plot_loss_comparison(history)
    print("\nAll plots saved to plots/ directory!")