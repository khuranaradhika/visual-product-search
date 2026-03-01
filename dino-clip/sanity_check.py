"""
Sanity check: take items FROM the catalog, query the index,
and see if they retrieve themselves at rank 1.

This isolates whether the problem is:
  A) The index/search (IVF clustering, nprobe too low)
  B) The model (embeddings don't capture visual similarity)
  C) The app (transforms, query pipeline mismatch)

Usage: python sanity_check.py
"""
import os, json, math
import numpy as np
import torch
import faiss
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from main import (
    Config, TwoTowerModel, ABODataset, collate_fn,
    load_faiss_index, query_index, parse_attributes,
)


def main():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    # --- Load both indexes ---
    print("=" * 60)
    print("LOADING INDEXES")
    print("=" * 60)

    # Fused index
    fused_index, fused_ids = load_faiss_index(cfg)
    if hasattr(fused_index, "nprobe"):
        print(f"  Fused index: IVF nlist={fused_index.nlist}, nprobe={fused_index.nprobe}")

    # Image-only index
    img_index_path = cfg.index_save_path.replace(".bin", "_imgonly.bin")
    img_ids_path = cfg.catalog_ids_path.replace(".json", "_imgonly.json")
    if os.path.exists(img_index_path):
        img_index = faiss.read_index(img_index_path)
        with open(img_ids_path) as f:
            img_ids = json.load(f)
        if hasattr(img_index, "nprobe"):
            print(f"  Image index: IVF nlist={img_index.nlist}, nprobe={img_index.nprobe}")
    else:
        img_index, img_ids = None, None
        print("  Image-only index not found, skipping")

    # --- Pick 10 random catalog items ---
    ds = ABODataset(cfg)
    np.random.seed(42)
    test_indices = np.random.choice(len(ds.samples), size=10, replace=False)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # --- Test 1: Brute-force cosine (no FAISS) ---
    print("\n" + "=" * 60)
    print("TEST 1: Brute-force image-only (no FAISS, no IVF)")
    print("  This tests if the model embeddings are good")
    print("=" * 60)

    # Build a small brute-force index from image embeddings
    print("  Encoding full catalog (image-only)...")
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,  # safe for this test
    )
    all_img_embs, all_ids_list = [], []
    with torch.no_grad():
        for imgs, texts, attr_lists, ids in tqdm(loader, desc="  Encoding"):
            imgs = imgs.to(cfg.device)
            embs = model.encode_image_only(imgs).cpu().numpy()
            all_img_embs.append(embs)
            all_ids_list.extend(ids)
    all_img_embs = np.vstack(all_img_embs).astype("float32")
    faiss.normalize_L2(all_img_embs)

    # Brute-force flat index
    d = all_img_embs.shape[1]
    bf_index = faiss.IndexFlatIP(d)
    bf_index.add(all_img_embs)

    self_hit_bf = 0
    for idx in test_indices:
        img_path, attr_text, item_id = ds.samples[idx]
        img = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(cfg.device)

        with torch.no_grad():
            emb = model.encode_image_only(img).cpu().numpy().astype("float32")
        faiss.normalize_L2(emb)

        scores, indices = bf_index.search(emb, 5)
        top5_ids = [all_ids_list[i] for i in indices[0]]
        rank1_match = "✅" if top5_ids[0] == item_id else "❌"
        in_top5 = "✅" if item_id in top5_ids else "❌"
        self_hit_bf += (top5_ids[0] == item_id)

        print(f"  Query: {item_id}")
        print(f"    Rank-1 match: {rank1_match}  |  In top-5: {in_top5}")
        print(f"    Top-5 scores: {[f'{s:.4f}' for s in scores[0]]}")
        print(f"    Top-5 IDs:    {top5_ids}")

    print(f"\n  Brute-force self-retrieval: {self_hit_bf}/10")

    # --- Test 2: IVF image-only index ---
    if img_index is not None:
        print("\n" + "=" * 60)
        print("TEST 2: IVF image-only index (saved index)")
        print("=" * 60)

        # Test with increasing nprobe
        for nprobe in [16, 64, 128, 256, 384]:
            if hasattr(img_index, "nprobe"):
                img_index.nprobe = nprobe

            self_hit = 0
            for idx in test_indices:
                img_path, attr_text, item_id = ds.samples[idx]
                img = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(cfg.device)

                with torch.no_grad():
                    emb = model.encode_image_only(img).cpu().numpy().astype("float32")
                faiss.normalize_L2(emb)

                scores, indices = img_index.search(emb, 5)
                top5_ids = [img_ids[i] for i in indices[0] if i >= 0]
                self_hit += (len(top5_ids) > 0 and top5_ids[0] == item_id)

            print(f"  nprobe={nprobe:4d} → self-retrieval: {self_hit}/10")

    # --- Test 3: Fused index ---
    print("\n" + "=" * 60)
    print("TEST 3: Fused index (original)")
    print("=" * 60)

    for nprobe in [16, 64, 128, 256, 384]:
        if hasattr(fused_index, "nprobe"):
            fused_index.nprobe = nprobe

        self_hit = 0
        for idx in test_indices:
            img_path, attr_text, item_id = ds.samples[idx]
            img = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0).to(cfg.device)
            attrs = parse_attributes(attr_text)

            with torch.no_grad():
                emb = model(img, [attrs]).cpu().numpy().astype("float32")
            faiss.normalize_L2(emb)

            scores, indices = fused_index.search(emb, 5)
            top5_ids = [fused_ids[i] for i in indices[0] if i >= 0]
            self_hit += (len(top5_ids) > 0 and top5_ids[0] == item_id)

        print(f"  nprobe={nprobe:4d} → self-retrieval: {self_hit}/10")

    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    print("If Test 1 gets 10/10 but Test 2 doesn't → IVF nprobe too low")
    print("If Test 1 fails too → model embeddings don't preserve identity")
    print("If Test 3 fails but Test 2 works → fused embeddings are the problem")


if __name__ == "__main__":
    main()