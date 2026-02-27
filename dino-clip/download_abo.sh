#!/bin/bash
# ============================================================
# Download the ABO dataset (listings metadata + catalog images)
# Run from your project root:  bash download_abo.sh
# ============================================================
# Requirements: AWS CLI  (brew install awscli)
# No AWS account needed — uses --no-sign-request
#
# Safe to interrupt and re-run: s3 sync skips already-downloaded files.
# ============================================================
set -e
DATA_DIR="data/abo"
mkdir -p "$DATA_DIR"

echo "=== 1/3  Downloading listings metadata (~68 MB compressed) ==="
aws s3 sync s3://amazon-berkeley-objects/listings/metadata/ \
    "$DATA_DIR/listings/metadata/" \
    --no-sign-request

echo ""
echo "=== 2/3  Downloading images metadata CSV ==="
aws s3 sync s3://amazon-berkeley-objects/images/metadata/ \
    "$DATA_DIR/images/metadata/" \
    --no-sign-request

echo ""
echo "=== 3/3  Downloading catalog images (~40 GB total) ==="
echo "    This is large! You can Ctrl-C and resume later — sync picks up where it left off."
echo "    Or download a subset with:  aws s3 sync ... --exclude '*' --include '0/*' "
echo ""
aws s3 sync s3://amazon-berkeley-objects/images/original/ \
    "$DATA_DIR/images/original/" \
    --no-sign-request

echo ""
echo "=== Done! ==="
echo "Expected layout:"
echo "  data/abo/listings/metadata/*.json.gz   (product listings, one JSON per line)"
echo "  data/abo/images/metadata/images.csv.gz (image_id → path mapping)"
echo "  data/abo/images/original/...           (actual JPG/PNG files)"