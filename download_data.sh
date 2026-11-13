#!/usr/bin/env bash
# Download the MASSIVE 1.0 dataset and the two frozen word-vector files used by main.ipynb.
#   1.0/data/en-US.jsonl              MASSIVE en-US (train / dev / test via its `partition` field)
#   embeddings/glove.6B.300d.txt      GloVe 6B, 300-d   (from glove.6B.zip, ~820 MB)
#   embeddings/crawl-300d-2M.vec      fastText Common Crawl, 300-d (from crawl-300d-2M.vec.zip, ~1.5 GB)
# Safe to re-run: files that already exist are skipped.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f 1.0/data/en-US.jsonl ]; then
  curl -fL https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz \
    -o amazon-massive-dataset-1.0.tar.gz
  tar -xzf amazon-massive-dataset-1.0.tar.gz 1.0/data/en-US.jsonl
  rm amazon-massive-dataset-1.0.tar.gz
fi

mkdir -p embeddings
if [ ! -f embeddings/glove.6B.300d.txt ]; then
  # Stanford's direct host first (nlp.stanford.edu only redirects there), then the Hugging Face mirror.
  curl -fL https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip -o embeddings/glove.6B.zip \
    || curl -fL https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip -o embeddings/glove.6B.zip
  unzip -o embeddings/glove.6B.zip glove.6B.300d.txt -d embeddings
  rm embeddings/glove.6B.zip
fi
if [ ! -f embeddings/crawl-300d-2M.vec ]; then
  curl -fL https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip \
    -o embeddings/crawl-300d-2M.vec.zip
  unzip -o embeddings/crawl-300d-2M.vec.zip -d embeddings
  rm embeddings/crawl-300d-2M.vec.zip
fi

echo "en-US records: $(grep -c '' 1.0/data/en-US.jsonl) (expected 16521)"
ls -lh embeddings
