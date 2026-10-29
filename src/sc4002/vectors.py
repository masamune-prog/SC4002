"""Pretrained GloVe 6B 300d word vectors as one frozen table.

Table rows = the vector keys of the train vocabulary (and their back-off parts) plus the
top-K most frequent GloVe words. K comes from GloVe's own frequency ranking, never from
dev/test, so the vocabulary is fit on train only. Unseen keys at dev/test time are looked
up with the same back-off rules against this fixed table.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import zipfile
from pathlib import Path

import numpy as np

from .data import ROOT

EMB_DIR = Path(os.environ.get("SC4002_EMB_DIR", ROOT / "embeddings"))
GLOVE_ZIP, GLOVE_MEMBER = EMB_DIR / "glove.6B.zip", "glove.6B.300d.txt"
DIM = 300
MAX_TOP_K = 100000
_SPLIT = re.compile(r"[-./&_']+|(?=n't)|(?='[a-z]+$)")
_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")


def backoff_keys(key: str) -> list[list[str]]:
    """Candidate look-ups in priority order; each candidate is averaged over its parts."""
    cands = [[key]]
    stripped = _PUNCT.sub("", key)
    if stripped and stripped != key:
        cands.append([stripped])
    if "." in key and key.replace(".", ""):
        cands.append([key.replace(".", "")])
    parts = [p for p in _SPLIT.split(key) if p]
    if len(parts) > 1:
        cands.append(parts)
    return cands


def load_store(train_keys: set[str]) -> dict:
    """Stream GloVe once, keeping the words these train keys could need plus the MAX_TOP_K
    most frequent words (GloVe files are ordered by frequency). Cached as .npz."""
    needed = set(train_keys) | {p for k in train_keys for c in backoff_keys(k) for p in c}
    tag = hashlib.md5("\n".join(sorted(needed)).encode()).hexdigest()[:10]
    cache = EMB_DIR / f"glove_store_{tag}_{MAX_TOP_K}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return dict(words=list(z["words"]), matrix=z["matrix"], top=list(z["top"]))
    words, vecs, top = [], [], []
    with zipfile.ZipFile(GLOVE_ZIP) as z, z.open(GLOVE_MEMBER) as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8", errors="ignore"):
            word, rest = line.rstrip("\n").split(" ", 1)
            in_top = len(top) < MAX_TOP_K
            if in_top:
                top.append(word)
            if in_top or word in needed:
                words.append(word)
                vecs.append(np.asarray(rest.split(" "), dtype=np.float32))
    matrix = np.stack(vecs)
    assert matrix.shape[1] == DIM
    np.savez(cache, words=np.array(words), matrix=matrix, top=np.array(top))
    return dict(words=words, matrix=matrix, top=top)


class VectorTable:
    """Frozen table (row 0 = zero vector) plus the back-off lookup used for every split."""

    def __init__(self, store: dict, train_keys: set[str], top_k: int, extra_rows: int | None = None):
        """extra_rows caps how many top-K rows beyond the train rows are kept (budget filling)."""
        idx = {w: i for i, w in enumerate(store["words"])}
        rows = []
        for k in sorted(train_keys):
            for cand in backoff_keys(k):
                hits = [p for p in cand if p in idx]
                if hits:
                    rows += hits
                    break
        self.n_train_rows = len(dict.fromkeys(rows))
        rows += store["top"][:top_k]
        self.words = list(dict.fromkeys(rows))
        if extra_rows is not None:
            self.words = self.words[: self.n_train_rows + max(0, extra_rows)]
        self.row = {w: i + 1 for i, w in enumerate(self.words)}
        mat = store["matrix"][[idx[w] for w in self.words]]
        self.matrix = np.concatenate([np.zeros((1, DIM), np.float32), mat])
        self._cache: dict[str, list[int]] = {}

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def lookup(self, key: str) -> list[int]:
        if key not in self._cache:
            out = []
            for cand in backoff_keys(key):
                out = [self.row[p] for p in cand if p in self.row]
                if out:
                    break
            self._cache[key] = out
        return self._cache[key]

    def coverage(self, keys) -> float:
        keys = list(keys)
        return 100.0 * sum(bool(self.lookup(k)) for k in keys) / max(len(keys), 1)
