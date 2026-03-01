"""
Build a FAISS index using only image embeddings (no text fusion).
This enables exact visual matching in the Streamlit app.

Usage:  python build_img_index.py
"""
import os, json, math
import numpy as np
import torch
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm
from main import (
    Config, TwoTowerModel, ABODataset, collate_fn,
)

def main():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    ds = ABODataset(cfg)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
    )

    all_embs, all_ids = [], []
    with torch.no_grad():
        for imgs, texts, attr_lists, ids in tqdm(loader, desc="Encoding images"):
            imgs = imgs.to(cfg.device)
            embs = model.encode_image_only(imgs).cpu().numpy()
            all_embs.append(embs)
            all_ids.extend(ids)

    catalog_embs = np.ascontiguousarray(
        np.vstack(all_embs), dtype="float32"
    )
    faiss.normalize_L2(catalog_embs)

    n, d = catalog_embs.shape
    if n < 10_000:
        index = faiss.IndexFlatIP(d)
        index.add(catalog_embs)
    else:
        nlist = min(int(math.sqrt(n)), 1024)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(catalog_embs)
        index.add(catalog_embs)
        index.nprobe = min(nlist // 4, 64)

    # Save next to the main index
    img_index_path = cfg.index_save_path.replace(".bin", "_imgonly.bin")
    img_ids_path = cfg.catalog_ids_path.replace(".json", "_imgonly.json")

    os.makedirs(os.path.dirname(img_index_path), exist_ok=True)
    faiss.write_index(index, img_index_path)
    with open(img_ids_path, "w") as f:
        json.dump(all_ids, f)

    print(f"[Done] Image-only index: {index.ntotal} vectors, dim={d}")
    print(f"       Saved to {img_index_path}")


if __name__ == "__main__":
    main()