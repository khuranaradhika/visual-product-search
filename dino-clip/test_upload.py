"""
Quick test: open the same image two ways and compare embeddings.
Simulates what happens when Streamlit re-reads an uploaded file.

Usage: python test_upload.py
"""
import io, os, json
import numpy as np
import torch
import faiss
from PIL import Image
from torchvision import transforms
from main import (
    Config, TwoTowerModel, ABODataset, load_faiss_index, parse_attributes,
)


def main():
    cfg = Config()
    model = TwoTowerModel(cfg).to(cfg.device)
    model.load_state_dict(
        torch.load(cfg.model_save_path, map_location=cfg.device, weights_only=True)
    )
    model.eval()

    # Load image-only index
    img_index_path = cfg.index_save_path.replace(".bin", "_imgonly.bin")
    img_ids_path = cfg.catalog_ids_path.replace(".json", "_imgonly.json")
    if os.path.exists(img_index_path):
        index = faiss.read_index(img_index_path)
        with open(img_ids_path) as f:
            catalog_ids = json.load(f)
        if hasattr(index, "nprobe"):
            index.nprobe = 384
    else:
        index, catalog_ids = load_faiss_index(cfg)
        if hasattr(index, "nprobe"):
            index.nprobe = 384

    ds = ABODataset(cfg)
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Pick 5 test items
    np.random.seed(123)
    test_indices = np.random.choice(len(ds.samples), size=5, replace=False)

    print("=" * 70)
    print("Comparing: disk read vs simulated upload (bytes round-trip)")
    print("=" * 70)

    for idx in test_indices:
        img_path, attr_text, item_id = ds.samples[idx]

        # Method 1: Direct from disk (how sanity_check.py does it)
        img_disk = Image.open(img_path).convert("RGB")

        # Method 2: Simulate Streamlit upload (read bytes, then open)
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        img_upload = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Method 3: JPEG re-compression (worst case upload simulation)
        buf = io.BytesIO()
        img_disk.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        img_recompressed = Image.open(buf).convert("RGB")

        t_disk = tfm(img_disk).unsqueeze(0).to(cfg.device)
        t_upload = tfm(img_upload).unsqueeze(0).to(cfg.device)
        t_recomp = tfm(img_recompressed).unsqueeze(0).to(cfg.device)

        with torch.no_grad():
            e_disk = model.encode_image_only(t_disk).cpu().numpy().astype("float32")
            e_upload = model.encode_image_only(t_upload).cpu().numpy().astype("float32")
            e_recomp = model.encode_image_only(t_recomp).cpu().numpy().astype("float32")

        faiss.normalize_L2(e_disk)
        faiss.normalize_L2(e_upload)
        faiss.normalize_L2(e_recomp)

        # Cosine similarities between methods
        cos_disk_upload = float((e_disk @ e_upload.T)[0, 0])
        cos_disk_recomp = float((e_disk @ e_recomp.T)[0, 0])

        # Search with each
        _, idx_disk = index.search(e_disk, 3)
        _, idx_upload = index.search(e_upload, 3)
        _, idx_recomp = index.search(e_recomp, 3)

        top1_disk = catalog_ids[idx_disk[0][0]]
        top1_upload = catalog_ids[idx_upload[0][0]]
        top1_recomp = catalog_ids[idx_recomp[0][0]]

        disk_ok = "✅" if top1_disk == item_id else "❌"
        upload_ok = "✅" if top1_upload == item_id else "❌"
        recomp_ok = "✅" if top1_recomp == item_id else "❌"

        print(f"\n  Item: {item_id}")
        print(f"    disk→upload cosine:     {cos_disk_upload:.6f}")
        print(f"    disk→recompressed cosine: {cos_disk_recomp:.6f}")
        print(f"    Retrieval: disk={disk_ok}  upload={upload_ok}  recomp={recomp_ok}")

    print("\n" + "=" * 70)
    print("If recomp fails but disk/upload pass → JPEG compression shifts embeddings")
    print("If all pass → the issue is in app.py logic, not image processing")
    print("If upload fails → PIL BytesIO handling differs")


if __name__ == "__main__":
    main()