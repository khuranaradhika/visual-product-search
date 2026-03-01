# Image Retrieval with CLIP on Amazon Berkeley Objects (ABO)

## Stages
  1. load_data  – parse ABO listings + images, build text descriptions
  2. embed      – fused embeddings: L2_norm(img_emb + txt_emb)
  3. index      – FAISS IndexFlatIP (exact cosine similarity)
  4. evaluate   – Precision@K, Recall@K, mAP@K, nDCG@K, MRR
  5. finetune   – symmetric contrastive loss on ABO image-text pairs
                  backbone cached → projection head only trains
  6. visualise  – matplotlib retrieval grid

## Modes (set MODE below):
  "pretrained"  – embed → index → evaluate (zero-shot)
  "finetuned"   – finetune → embed → index → evaluate
