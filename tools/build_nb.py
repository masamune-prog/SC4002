"""Builds main.ipynb from the cell sources below.

Usage (from anywhere): python tools/build_nb.py [out.ipynb]   (default: <repo>/main.ipynb)
This only writes the cells; execute the notebook afterwards so its outputs are produced by a real run.
"""
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(new_code_cell(s.strip("\n")))

# ----------------------------------------------------------------------------- 0. setup
md(r'''
# SC4002 Group Assignment: Joint Intent Classification and Slot Filling on MASSIVE (en-US)

**Stage 1: independent baseline.** A joint BiLSTM model that predicts, for each utterance, one of 60 intents
and the set of `[slot type : slot span]` pairs (55 slot types).

Notebook layout:
0. Environment and seeds
1. Load and verify the dataset (official splits)
2. Data checks
3. Preprocessing (tokens, BIO tags, vocabularies, tensors)
4. Shared evaluation functions and unit tests
5. Pretrained word vectors (GloVe 6B 300d)
6. Stage-1 model: joint BiLSTM with character CNN and CRF
7. Model checks: parameter budget, CRF against brute force, overfitting a small batch
''')

code(r'''
import os
# cuBLAS needs this workspace setting for deterministic kernels; it must be set before torch is imported.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import io
import itertools
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Python  {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | cuDNN {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {DEVICE} | seed: {SEED}")
''')

# ----------------------------------------------------------------------------- 1. data
md(r'''
## 1. Load and verify the dataset

MASSIVE 1.0 is downloaded from the official archive and `1.0/data/en-US.jsonl` is extracted. Records are split by
their `partition` field, and no example is added, removed or moved between splits.
''')

code(r'''
DATA_DIR = Path("data")
ARCHIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
ARCHIVE_PATH = DATA_DIR / "amazon-massive-dataset-1.0.tar.gz"
MEMBER = "1.0/data/en-US.jsonl"
JSONL_PATH = DATA_DIR / MEMBER

if not JSONL_PATH.exists():
    DATA_DIR.mkdir(exist_ok=True)
    if not ARCHIVE_PATH.exists():
        print(f"Downloading {ARCHIVE_URL}")
        urllib.request.urlretrieve(ARCHIVE_URL, ARCHIVE_PATH)
    with tarfile.open(ARCHIVE_PATH) as tar:
        tar.extract(MEMBER, DATA_DIR, filter="data")
print(f"Using {JSONL_PATH}")

with open(JSONL_PATH, encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

splits = {p: [r for r in records if r["partition"] == p] for p in ("train", "dev", "test")}
for name, rs in splits.items():
    print(f"{name:5s}: {len(rs):6,d} examples")
print(f"total: {len(records):6,d} examples")

assert len(splits["train"]) == 11_514
assert len(splits["dev"]) == 2_033
assert len(splits["test"]) == 2_974
assert len(records) == 16_521 == sum(len(rs) for rs in splits.values())
print("Split sizes verified.")
''')

# ----------------------------------------------------------------------------- 2. checks
md(r'''
## 2. Data checks

The only model input is `utt`. Labels come from `intent` and `annot_utt` (`... [type : span] ...`).
Checks:
- Removing the bracket annotation from `annot_utt` gives back `utt` exactly, for every example.
- The text is lowercase and every slot span lies on whitespace boundaries, so whitespace tokenisation of `utt`
  lines up with the slot labels.
''')

code(r'''
SLOT_PATTERN = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")

def strip_annotation(annot_utt):
    return SLOT_PATTERN.sub(lambda m: m.group(2), annot_utt)

mismatches = [r for r in records if strip_annotation(r["annot_utt"]) != r["utt"]]
print(f"strip(annot_utt) != utt: {len(mismatches)} examples")
assert not mismatches

leftover = [r for r in records if re.search(r"[\[\]]", SLOT_PATTERN.sub("", r["annot_utt"]))]
assert not leftover, "unparsed bracket in annot_utt"
assert not any(c.isupper() for r in records for c in r["utt"]), "unexpected uppercase"

punct = Counter(c for r in records for c in r["utt"] if not c.isalnum() and c != " ")
print("Punctuation characters in utt:", dict(punct.most_common()))
lengths = [len(r["utt"].split()) for r in records]
print(f"Utterance length (tokens): mean {np.mean(lengths):.2f}, max {max(lengths)}")

rows = []
for name, rs in splits.items():
    slot_types = Counter(m.group(1) for r in rs for m in SLOT_PATTERN.finditer(r["annot_utt"]))
    rows.append({
        "split": name,
        "examples": len(rs),
        "intents": len({r["intent"] for r in rs}),
        "slot types": len(slot_types),
        "slot spans": sum(slot_types.values()),
        "utts without slots": sum(not SLOT_PATTERN.search(r["annot_utt"]) for r in rs),
    })
display(pd.DataFrame(rows).set_index("split"))
''')

# ----------------------------------------------------------------------------- 3. preprocessing
md(r'''
## 3. Preprocessing

- **Tokens:** `utt.split()`. The data is already lowercase, so no normalisation is applied.
- **Slot labels:** `annot_utt` is converted to one BIO tag per token. The first token of a span gets `B-type`, the
  remaining span tokens get `I-type`, and every other token gets `O`. All 55 slot types appear in train, which gives
  111 tags.
- **Intent labels:** an index into the 60 intents seen in train.
- **Vocabularies** (word, character, intent, tag) are built from the **train split only**. Words and characters
  not in the train vocabulary map to `<<UNK>>`. Special-token names use double angle brackets so they cannot collide
  with the literal token `<empty>` that appears in the data.
''')

code(r'''
def parse_gold_pairs(annot_utt):
    """Set of (slot type, span text) pairs, parsed directly from annot_utt."""
    return {(m.group(1), m.group(2)) for m in SLOT_PATTERN.finditer(annot_utt)}

def bio_tags(annot_utt):
    """Whitespace tokens of the utterance and one BIO tag per token."""
    tokens, tags, pos = [], [], 0
    for m in SLOT_PATTERN.finditer(annot_utt):
        outside = annot_utt[pos:m.start()].split()
        tokens += outside
        tags += ["O"] * len(outside)
        span = m.group(2).split()
        tokens += span
        tags += [f"B-{m.group(1)}"] + [f"I-{m.group(1)}"] * (len(span) - 1)
        pos = m.end()
    outside = annot_utt[pos:].split()
    tokens += outside
    tags += ["O"] * len(outside)
    return tokens, tags

def tags_to_pairs(tokens, tags):
    """Decode BIO tags into a set of (slot type, span text) pairs.

    Part of the model's output layer, not the metric. B-x always opens a new span; I-x extends an open span of
    the same type, and otherwise (orphan I-x after O or after a different type) opens a new span of type x.
    """
    spans, cur_type, cur_tokens = [], None, []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            prefix, slot = "O", None
        else:
            prefix, slot = tag.split("-", 1)
        if prefix == "I" and slot == cur_type:
            cur_tokens.append(tok)
            continue
        if cur_type is not None:
            spans.append((cur_type, " ".join(cur_tokens)))
        cur_type, cur_tokens = (slot, [tok]) if prefix in ("B", "I") else (None, [])
    if cur_type is not None:
        spans.append((cur_type, " ".join(cur_tokens)))
    return set(spans)

for r in records:
    tokens, _ = bio_tags(r["annot_utt"])
    assert tokens == r["utt"].split(), r["id"]
print("BIO tokenisation matches utt.split() for all", len(records), "examples.")

example = splits["train"][0]
print("\nannot_utt:", example["annot_utt"])
for tok, tag in zip(*bio_tags(example["annot_utt"])):
    print(f"  {tok:12s} {tag}")
''')

code(r'''
PAD, UNK = "<<PAD>>", "<<UNK>>"

train_tokens = [bio_tags(r["annot_utt"])[0] for r in splits["train"]]
word_counts = Counter(w for toks in train_tokens for w in toks)
char_counts = Counter(c for w in word_counts for c in w)
assert PAD not in word_counts and UNK not in word_counts

itos_word = [PAD, UNK] + sorted(word_counts)
itos_char = [PAD, UNK] + sorted(char_counts)
itos_intent = sorted({r["intent"] for r in splits["train"]})
train_slot_types = sorted({t for r in splits["train"] for t, _ in parse_gold_pairs(r["annot_utt"])})
itos_tag = ["O"] + [f"{p}-{t}" for t in train_slot_types for p in ("B", "I")]

stoi_word = {w: i for i, w in enumerate(itos_word)}
stoi_char = {c: i for i, c in enumerate(itos_char)}
stoi_intent = {x: i for i, x in enumerate(itos_intent)}
stoi_tag = {t: i for i, t in enumerate(itos_tag)}
PAD_ID, UNK_ID = stoi_word[PAD], stoi_word[UNK]
singletons = {stoi_word[w] for w, c in word_counts.items() if c == 1}

print(f"word vocab: {len(itos_word):,} ({len(word_counts):,} train types + PAD/UNK), singletons: {len(singletons):,}")
print(f"char vocab: {len(itos_char)} | intents: {len(itos_intent)} | slot types: {len(train_slot_types)} | tags: {len(itos_tag)}")
assert len(itos_intent) == 60 and len(train_slot_types) == 55 and len(itos_tag) == 111

for name in ("dev", "test"):
    toks = [w for r in splits[name] for w in r["utt"].split()]
    oov = sum(w not in stoi_word for w in toks)
    print(f"{name:4s} OOV vs train vocab: {oov}/{len(toks)} tokens ({100 * oov / len(toks):.2f}%)")
    assert {r["intent"] for r in splits[name]} <= set(stoi_intent)
    assert all(t in stoi_tag for r in splits[name] for t in bio_tags(r["annot_utt"])[1])
''')

code(r'''
def encode(record):
    """Model input from utt only; labels from intent and annot_utt."""
    tokens = record["utt"].split()
    _, tags = bio_tags(record["annot_utt"])
    return {
        "tokens": tokens,
        "word_ids": [stoi_word.get(w, UNK_ID) for w in tokens],
        "char_ids": [[stoi_char.get(c, stoi_char[UNK]) for c in w] for w in tokens],
        "tag_ids": [stoi_tag[t] for t in tags],
        "intent_id": stoi_intent[record["intent"]],
        "annot_utt": record["annot_utt"],
        "intent": record["intent"],
    }

encoded = {name: [encode(r) for r in rs] for name, rs in splits.items()}

def collate(batch):
    """Pad a list of encoded examples into tensors: words [B,T], chars [B,T,L], tags [B,T], mask [B,T], intents [B]."""
    B = len(batch)
    T = max(len(ex["word_ids"]) for ex in batch)
    L = max(len(c) for ex in batch for c in ex["char_ids"])
    words = torch.full((B, T), PAD_ID, dtype=torch.long)
    chars = torch.full((B, T, L), stoi_char[PAD], dtype=torch.long)
    tags = torch.zeros((B, T), dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for i, ex in enumerate(batch):
        n = len(ex["word_ids"])
        words[i, :n] = torch.tensor(ex["word_ids"])
        tags[i, :n] = torch.tensor(ex["tag_ids"])
        mask[i, :n] = True
        for j, c in enumerate(ex["char_ids"]):
            chars[i, j, :len(c)] = torch.tensor(c)
    intents = torch.tensor([ex["intent_id"] for ex in batch])
    return {"words": words, "chars": chars, "tags": tags, "mask": mask, "intents": intents}

demo = collate(encoded["train"][:4])
print({k: tuple(v.shape) for k, v in demo.items()})
''')

# ----------------------------------------------------------------------------- 4. evaluator
md(r'''
## 4. Shared evaluation functions

Every experiment in this notebook is scored by the same function, `evaluate`:
- **Intent accuracy** = 100 × correct / total.
- **Slot micro-F1**: for each utterance, the predicted set of `(slot type, span text)` pairs is compared with the
  gold set parsed directly from `annot_utt`. A pair is a true positive only when both type and span text match.
  TP/FP/FN are summed over all utterances, and F1 = 100 × 2TP / (2TP + FP + FN). Pairs are sets, so a pair repeated
  within one utterance counts once.
- **Combined** = (intent accuracy + slot F1) / 2, used only for checkpoint selection.

Turning predicted BIO tags into pairs (`tags_to_pairs`, above) is part of the model, not the metric. The evaluator
only sees pair sets. Scores are kept at full precision and rounded to two decimals only for display.
''')

code(r'''
def evaluate(gold_intents, pred_intents, gold_annot_utts, pred_pairs):
    """Shared scorer. pred_pairs is a list of sets of (slot type, span text), one per utterance."""
    n = len(gold_intents)
    assert n == len(pred_intents) == len(gold_annot_utts) == len(pred_pairs) and n > 0
    correct = sum(g == p for g, p in zip(gold_intents, pred_intents))
    tp = fp = fn = 0
    for annot, pred in zip(gold_annot_utts, pred_pairs):
        gold, pred = parse_gold_pairs(annot), set(pred)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    intent_acc = 100.0 * correct / n
    denom = 2 * tp + fp + fn
    slot_f1 = 100.0 * 2 * tp / denom if denom else 100.0
    return {"intent_acc": intent_acc, "slot_f1": slot_f1, "combined": (intent_acc + slot_f1) / 2,
            "n": n, "intent_correct": correct, "tp": tp, "fp": fp, "fn": fn}

def format_scores(s):
    return f"intent acc {s['intent_acc']:.2f} | slot F1 {s['slot_f1']:.2f} | combined {s['combined']:.2f}"
''')

md(r'''
### Evaluator unit tests

Hand-built cases with known answers. Gold always comes from an `annot_utt` string.
''')

code(r'''
GOLD = "wake me up at [time : five am] [date : this week]"
TOKENS = "wake me up at five am this week".split()

def slot_counts(pred, gold=GOLD):
    s = evaluate(["x"], ["x"], [gold], [pred])
    return s["tp"], s["fp"], s["fn"], s["slot_f1"]

# perfect match
assert slot_counts({("time", "five am"), ("date", "this week")}) == (2, 0, 0, 100.0)
# wrong type: span text right, type wrong
assert slot_counts({("date", "five am"), ("date", "this week")}) == (1, 1, 1, 50.0)
# wrong boundary: type right, span too short / too long
assert slot_counts({("time", "five"), ("date", "this week")}) == (1, 1, 1, 50.0)
assert slot_counts({("time", "at five am"), ("date", "this week")}) == (1, 1, 1, 50.0)
# extra span
assert slot_counts({("time", "five am"), ("date", "this week"), ("person", "me")}) == (2, 1, 0, 80.0)
# missing span
tp, fp, fn, f1 = slot_counts({("time", "five am")})
assert (tp, fp, fn) == (1, 0, 1) and abs(f1 - 200 / 3) < 1e-9
# no gold slots and no predictions: nothing to count
assert slot_counts(set(), gold="what's up") == (0, 0, 0, 100.0)
# a pair repeated in gold counts once (set semantics)
assert slot_counts({("person", "david")}, gold="email [person : david] about it [person : david]") == (1, 0, 0, 100.0)
# span text containing punctuation
assert slot_counts({("business_name", "domino's")}, gold="order from [business_name : domino's]") == (1, 0, 0, 100.0)

# malformed tag sequences -> pairs (decoding rule: orphan I-x opens a new span)
O = "O"
assert tags_to_pairs(TOKENS, [O, O, O, O, "B-time", "I-time", "B-date", "I-date"]) == {("time", "five am"), ("date", "this week")}
assert tags_to_pairs(TOKENS, [O, O, O, O, "I-time", "I-time", "I-date", "I-date"]) == {("time", "five am"), ("date", "this week")}
assert tags_to_pairs(TOKENS, [O, O, O, O, "B-time", "I-date", O, O]) == {("time", "five"), ("date", "am")}
assert tags_to_pairs(TOKENS, [O, O, O, O, "I-time", O, "I-time", O]) == {("time", "five"), ("time", "this")}
assert tags_to_pairs(TOKENS, [O, O, O, O, "B-time", "B-time", O, O]) == {("time", "five"), ("time", "am")}
assert tags_to_pairs(TOKENS, [O] * 8) == set()
assert tags_to_pairs(TOKENS, [O] * 6 + ["B-date", "I-date"]) == {("date", "this week")}  # span at end of utterance
# malformed tags scored end to end: orphan I- recovers the right spans
assert slot_counts(tags_to_pairs(TOKENS, [O, O, O, O, "I-time", "I-time", "I-date", "I-date"])) == (2, 0, 0, 100.0)

# pooling across utterances: utt1 TP=1 FP=1 FN=0, utt2 TP=0 FP=0 FN=2 -> TP=1 FP=1 FN=2 -> F1 = 200/5 = 40
s = evaluate(["alarm_set", "email_sendemail"], ["alarm_set", "alarm_set"],
             [GOLD.replace(" [date : this week]", " this week"), "email [person : james] about [event_name : lunch]"],
             [{("time", "five am"), ("person", "me")}, set()])
assert (s["intent_correct"], s["tp"], s["fp"], s["fn"]) == (1, 1, 1, 2)
assert s["intent_acc"] == 50.0 and abs(s["slot_f1"] - 40.0) < 1e-9 and abs(s["combined"] - 45.0) < 1e-9
print("All evaluator unit tests passed.")
''')

code(r'''
# Round trip on the real data: gold tags -> decoded pairs must reproduce the annot_utt pairs exactly (100.00).
for name, exs in encoded.items():
    pairs = [tags_to_pairs(ex["tokens"], [itos_tag[t] for t in ex["tag_ids"]]) for ex in exs]
    golds = [ex["intent"] for ex in exs]
    s = evaluate(golds, golds, [ex["annot_utt"] for ex in exs], pairs)
    print(f"{name:5s} gold-tag round trip: {format_scores(s)} (TP={s['tp']}, FP={s['fp']}, FN={s['fn']})")
    assert s["fp"] == 0 and s["fn"] == 0 and s["slot_f1"] == 100.0
''')

# ----------------------------------------------------------------------------- 5. GloVe
md(r'''
## 5. Pretrained word vectors: GloVe 6B 300d

GloVe 6B is lowercase, which matches the data. The notebook downloads it from Stanford if it is missing. Only
vectors for **train-vocabulary** words are kept, so dev/test words never enter the vocabulary and stay `<<UNK>>`.

Lookup for each train word:
1. exact match;
2. otherwise the word with its periods removed, for spelled-out abbreviations such as `m.` → `m`, `p.` → `p`;
3. otherwise a random vector with the same standard deviation as the GloVe vectors that were found. `<<UNK>>` is
   also random and is learned during training through word dropout; `<<PAD>>` is all zeros.
''')

code(r'''
EMB_DIR = Path("embeddings")
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_ZIP = EMB_DIR / "glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"
EMB_DIM = 300

if not GLOVE_ZIP.exists():
    EMB_DIR.mkdir(exist_ok=True)
    print(f"Downloading {GLOVE_URL} (~820 MB)")
    urllib.request.urlretrieve(GLOVE_URL, GLOVE_ZIP)

train_words = itos_word[2:]
strip_periods = lambda w: w.replace(".", "")
wanted = set(train_words) | {strip_periods(w) for w in train_words}
glove = {}
with zipfile.ZipFile(GLOVE_ZIP) as zf, zf.open(GLOVE_MEMBER) as f:
    for line in io.TextIOWrapper(f, encoding="utf-8"):
        word, rest = line.rstrip("\n").split(" ", 1)
        if word in wanted:
            glove[word] = np.asarray(rest.split(), dtype=np.float32)
assert all(v.shape == (EMB_DIM,) for v in glove.values())

source = {}
for w in train_words:
    if w in glove:
        source[w] = "exact"
    elif strip_periods(w) and strip_periods(w) in glove:
        source[w] = "period-stripped"
    else:
        source[w] = "none"

n_types = len(train_words)
n_tokens = sum(word_counts.values())
for kind in ("exact", "period-stripped", "none"):
    types = [w for w in train_words if source[w] == kind]
    toks = sum(word_counts[w] for w in types)
    print(f"{kind:16s}: {len(types):5,d} / {n_types:,} train types ({100 * len(types) / n_types:5.2f}%) | "
          f"{toks:6,d} / {n_tokens:,} train tokens ({100 * toks / n_tokens:5.2f}%)")
print("period-stripped examples:", [w for w in train_words if source[w] == "period-stripped"][:12])
missing = sorted((w for w in train_words if source[w] == "none"), key=lambda w: (-word_counts[w], w))
print(f"\nNo GloVe vector, most frequent first ({len(missing)} words; train count in brackets):")
print(", ".join(f"{w} ({word_counts[w]})" for w in missing[:20]))
''')

code(r'''
found = np.stack([glove[w] for w in glove])
init_rng = np.random.default_rng(SEED)
emb_std = float(found.std())

emb_matrix = np.zeros((len(itos_word), EMB_DIM), dtype=np.float32)       # row 0 = <<PAD>> stays zero
emb_matrix[UNK_ID] = init_rng.normal(0.0, emb_std, EMB_DIM)
for w in train_words:
    i = stoi_word[w]
    if source[w] == "exact":
        emb_matrix[i] = glove[w]
    elif source[w] == "period-stripped":
        emb_matrix[i] = glove[strip_periods(w)]
    else:
        emb_matrix[i] = init_rng.normal(0.0, emb_std, EMB_DIM)
emb_matrix = torch.from_numpy(emb_matrix)
del glove, found
print(f"embedding matrix: {tuple(emb_matrix.shape)}, GloVe std used for random rows: {emb_std:.4f}")
''')

# ----------------------------------------------------------------------------- 6. model
md(r'''
## 6. Stage-1 model: joint BiLSTM with character CNN and CRF

**Inputs.** For each utterance, from `utt` only: word ids `[B, T]`, character ids `[B, T, L]`, and a padding mask.

**Architecture.** One shared encoder feeds two heads:
1. **Word embedding** (300-d): one row per train-vocabulary word, initialised from GloVe (Section 5). The rows are
   fine-tuned by default (`freeze_emb=False`); the frozen-GloVe variant is an ablation.
2. **Character CNN** (`use_char`): 50-d character embeddings → Conv1d (100 filters, width 3) → max-pool over the
   characters of each word → tanh, giving a 100-d feature per word. It carries spelling information, which helps
   with words that map to `<<UNK>>` (3.3% of dev tokens).
3. **Encoder**: [word ; char] (400-d) → dropout 0.5 → 2-layer BiLSTM (256 per direction, dropout 0.5 between
   layers) over packed sequences. Output: one 512-d state per token.
4. **Intent head**: masked mean ⊕ masked max over the token states (1024-d) → dropout → Linear(256) → ReLU →
   dropout → Linear(60). Pooling happens *before* dropout. Max-pooling over dropout-scaled states caused a
   train/eval mismatch, which the overfitting check in 7.3 caught.
5. **Slot head**: dropout → Linear(512 → 111) emission scores per token, then either
   - a **linear-chain CRF** (`use_crf=True`): a learned 111×111 transition matrix plus start/end scores, decoded
     with Viterbi to the best whole tag sequence; or
   - **softmax** (`use_crf=False`): an independent argmax per token.

**Outputs.** One intent label, plus BIO tags that `tags_to_pairs` turns into the set of `(slot type, span text)`
pairs.

**Loss.** L = CE(intent) + slot loss, averaged over the utterances in the batch. Slot loss per utterance:
- with the CRF, the negative log-likelihood of the gold tag sequence, −log p(y | x), computed with the forward
  algorithm;
- with softmax, the per-token cross-entropy summed over the utterance's tokens, which puts it on the same
  per-utterance scale as the CRF loss.

**Optimizer and training** (used in Section 8): Adam (lr 1e-3), batch size 32, gradient-norm clipping at 5.0,
40 epochs. Word dropout during training replaces words seen only once in train with `<<UNK>>` at p = 0.5, so the
`<<UNK>>` vector is learned.
''')

code(r'''
class CharCNN(nn.Module):
    """Character embeddings -> Conv1d -> max over characters -> tanh, one vector per word."""

    def __init__(self, n_chars, char_dim, n_filters, width, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(char_dim, n_filters, width, padding=width // 2)

    def forward(self, chars):                                   # chars: [B, T, L]
        B, T, L = chars.shape
        flat = chars.reshape(B * T, L)
        h = self.conv(self.emb(flat).transpose(1, 2))           # [B*T, F, L]
        pad = (flat == self.pad_id).unsqueeze(1)
        h = h.masked_fill(pad, float("-inf")).max(dim=2).values  # [B*T, F]
        h = h.masked_fill(pad.all(dim=2), 0.0)                  # padded words have no characters
        return torch.tanh(h).reshape(B, T, -1)


class CRF(nn.Module):
    """Linear-chain CRF. transitions[i, j] is the score of moving from tag i to tag j."""

    def __init__(self, n_tags):
        super().__init__()
        self.transitions = nn.Parameter(torch.empty(n_tags, n_tags).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.empty(n_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(n_tags).uniform_(-0.1, 0.1))

    def path_score(self, emissions, tags, mask):                 # -> [B]
        B, T, _ = emissions.shape
        emit = emissions.gather(2, tags.unsqueeze(2)).squeeze(2)  # [B, T]
        score = self.start[tags[:, 0]] + emit[:, 0]
        for t in range(1, T):
            step = self.transitions[tags[:, t - 1], tags[:, t]] + emit[:, t]
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        return score + self.end[last]

    def log_partition(self, emissions, mask):                    # forward algorithm -> [B]
        alpha = self.start + emissions[:, 0]                      # [B, K]
        for t in range(1, emissions.size(1)):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.transitions + emissions[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end, dim=1)

    def nll(self, emissions, tags, mask):                        # -log p(tags | x) -> [B]
        return self.log_partition(emissions, mask) - self.path_score(emissions, tags, mask)

    @torch.no_grad()
    def viterbi(self, emissions, mask):                          # -> list of tag-id lists
        score = self.start + emissions[:, 0]
        backpointers = []
        for t in range(1, emissions.size(1)):
            best, idx = (score.unsqueeze(2) + self.transitions).max(dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), best + emissions[:, t], score)
            backpointers.append(idx)
        score = score + self.end
        lengths = mask.sum(1).tolist()
        last = score.argmax(1).tolist()
        backpointers = torch.stack(backpointers, 1).cpu() if backpointers else None
        paths = []
        for b, n in enumerate(lengths):
            path = [last[b]]
            for t in range(n - 2, -1, -1):
                path.append(int(backpointers[b, t, path[-1]]))
            paths.append(path[::-1])
        return paths


class JointBiLSTM(nn.Module):
    """Shared BiLSTM encoder with an intent head (pooled states) and a slot head (CRF or softmax)."""

    def __init__(self, cfg, n_words, n_chars, n_intents, n_tags, word_pad_id, char_pad_id):
        super().__init__()
        self.cfg = cfg
        self.word_emb = nn.Embedding(n_words, cfg["emb_dim"], padding_idx=word_pad_id)
        self.word_emb.weight.requires_grad = not cfg["freeze_emb"]
        in_dim = cfg["emb_dim"]
        self.char_cnn = None
        if cfg["use_char"]:
            self.char_cnn = CharCNN(n_chars, cfg["char_dim"], cfg["char_filters"], cfg["char_width"], char_pad_id)
            in_dim += cfg["char_filters"]
        self.dropout = nn.Dropout(cfg["dropout"])
        self.lstm = nn.LSTM(in_dim, cfg["hidden"], num_layers=cfg["layers"], bidirectional=True,
                            batch_first=True, dropout=cfg["dropout"] if cfg["layers"] > 1 else 0.0)
        enc_dim = 2 * cfg["hidden"]
        self.intent_head = nn.Sequential(
            nn.Linear(2 * enc_dim, cfg["intent_hidden"]), nn.ReLU(), nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["intent_hidden"], n_intents))
        self.emission = nn.Linear(enc_dim, n_tags)
        self.crf = CRF(n_tags) if cfg["use_crf"] else None

    def forward(self, words, chars, mask):
        x = self.word_emb(words)
        if self.char_cnn is not None:
            x = torch.cat([x, self.char_cnn(chars)], dim=-1)
        x = self.dropout(x)
        lengths = mask.sum(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=words.size(1))  # [B, T, 2H]
        # Pool before dropout: max-pooling over dropout-scaled states picks inflated values in training that do not
        # occur in eval mode (the overfitting check in 7.3 caught this as a train/eval intent-accuracy gap).
        m = mask.unsqueeze(2)
        mean = (h * m).sum(1) / m.sum(1)
        mx = h.masked_fill(~m, float("-inf")).max(1).values
        intent_logits = self.intent_head(self.dropout(torch.cat([mean, mx], dim=-1)))
        return intent_logits, self.emission(self.dropout(h))

    def losses(self, intent_logits, emissions, batch):
        """Per-utterance loss [B] = intent CE + slot loss (CRF NLL, or token CE summed over the utterance)."""
        intent_loss = F.cross_entropy(intent_logits, batch["intents"], reduction="none")
        if self.crf is not None:
            slot_loss = self.crf.nll(emissions, batch["tags"], batch["mask"])
        else:
            token_ce = F.cross_entropy(emissions.transpose(1, 2), batch["tags"], reduction="none")
            slot_loss = (token_ce * batch["mask"]).sum(1)
        return intent_loss + slot_loss

    def decode_tags(self, emissions, mask):
        if self.crf is not None:
            return self.crf.viterbi(emissions, mask)
        best = emissions.argmax(-1).cpu()
        return [best[b, :n].tolist() for b, n in enumerate(mask.sum(1).tolist())]
''')

code(r'''
MODEL_CONFIG = {
    "use_char": True, "use_crf": True, "freeze_emb": False,
    "emb_dim": EMB_DIM, "char_dim": 50, "char_filters": 100, "char_width": 3,
    "hidden": 256, "layers": 2, "intent_hidden": 256, "dropout": 0.5,
}

def build_model(cfg, word_vectors=None):
    """Build the model from a config. word_vectors (GloVe init) is only needed for a fresh model, not a reload."""
    model = JointBiLSTM(cfg, len(itos_word), len(itos_char), len(itos_intent), len(itos_tag),
                        word_pad_id=PAD_ID, char_pad_id=stoi_char[PAD])
    if word_vectors is not None:
        with torch.no_grad():
            model.word_emb.weight.copy_(word_vectors)
    return model.to(DEVICE)

def to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}

@torch.no_grad()
def predict(model, examples, batch_size=256):
    """Run the model on encoded examples. Returns intent labels, slot-pair sets and the mean per-utterance loss."""
    model.eval()
    intents, pairs, loss_sum = [], [], 0.0
    for i in range(0, len(examples), batch_size):
        chunk = examples[i:i + batch_size]
        batch = to_device(collate(chunk))
        intent_logits, emissions = model(batch["words"], batch["chars"], batch["mask"])
        loss_sum += model.losses(intent_logits, emissions, batch).sum().item()
        intents += [itos_intent[j] for j in intent_logits.argmax(-1).tolist()]
        for ex, tag_ids in zip(chunk, model.decode_tags(emissions, batch["mask"])):
            pairs.append(tags_to_pairs(ex["tokens"], [itos_tag[t] for t in tag_ids]))
    return intents, pairs, loss_sum / len(examples)

def evaluate_model(model, examples, batch_size=256):
    """Predict, then score with the shared evaluate(). Also returns the mean loss."""
    intents, pairs, loss = predict(model, examples, batch_size)
    scores = evaluate([ex["intent"] for ex in examples], intents, [ex["annot_utt"] for ex in examples], pairs)
    return {**scores, "loss": loss}
''')

# ----------------------------------------------------------------------------- 7. checks
md(r'''
## 7. Model checks

### 7.1 Parameter budget

Total counts include every parameter in the pipeline, including the word-embedding matrix even when it is frozen.
Every variant must have at most 20M parameters.
''')

code(r'''
PARAM_LIMIT = 20_000_000
VARIANTS = {
    "full (char + CRF)":          {},
    "no char":                    {"use_char": False},
    "no CRF (softmax)":           {"use_crf": False},
    "plain BiLSTM-softmax":       {"use_char": False, "use_crf": False},
    "full, frozen GloVe":         {"freeze_emb": True},
}

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

rows = []
for name, overrides in VARIANTS.items():
    m = build_model({**MODEL_CONFIG, **overrides}, emb_matrix)
    total, trainable = count_params(m)
    assert total <= PARAM_LIMIT, (name, total)
    rows.append({"variant": name, "total params": f"{total:,}", "trainable params": f"{trainable:,}",
                 "<= 20M": total <= PARAM_LIMIT})
    del m
display(pd.DataFrame(rows).set_index("variant"))

m = build_model(MODEL_CONFIG, emb_matrix)
print("Stage-1 model (full), parameters by component:")
for name, child in m.named_children():
    n = sum(p.numel() for p in child.parameters())
    if n:
        print(f"  {name:12s} {n:12,d}")
total, trainable = count_params(m)
print(f"  {'TOTAL':12s} {total:12,d}  (trainable {trainable:,}) -> within 20M: {total <= PARAM_LIMIT}")
del m
''')

md(r'''
### 7.2 CRF correctness against brute force

The CRF is written by hand, so it is checked against a direct enumeration. On a small random problem (4 tags,
padded batch with lengths 4, 2, 3 and 1), every possible tag sequence is scored independently in float64 Python
from the CRF's own parameters. Assertions:
- the forward-algorithm log-partition equals the log-sum-exp over all sequences (within 1e-5);
- the gold-path score equals the brute-force score of that sequence;
- Viterbi returns the brute-force highest-scoring sequence;
- padding has no effect: each sequence scored alone, without padding, gives the same results.
''')

code(r'''
def brute_force(crf, emissions, length):
    """Score every tag sequence of the given length directly from the CRF parameters (float64)."""
    trans = crf.transitions.detach().double().cpu()
    start, end = crf.start.detach().double().cpu(), crf.end.detach().double().cpu()
    emit = emissions.detach().double().cpu()
    K = trans.size(0)
    scores = {}
    for seq in itertools.product(range(K), repeat=length):
        s = start[seq[0]] + end[seq[-1]] + sum(emit[t, seq[t]] for t in range(length))
        s = s + sum(trans[seq[t - 1], seq[t]] for t in range(1, length))
        scores[seq] = float(s)
    log_z = float(torch.logsumexp(torch.tensor(list(scores.values()), dtype=torch.float64), 0))
    return scores, log_z

set_seed(SEED)
K, lengths = 4, [4, 2, 3, 1]
T = max(lengths)
crf = CRF(K).to(DEVICE)
with torch.no_grad():                                           # larger-than-init parameters make the test harder
    for p in crf.parameters():
        p.normal_(0.0, 1.0)
emissions = torch.randn(len(lengths), T, K, device=DEVICE)
mask = torch.tensor([[t < n for t in range(T)] for n in lengths], device=DEVICE)
tags = torch.randint(0, K, (len(lengths), T), device=DEVICE) * mask

log_z = crf.log_partition(emissions, mask)
gold = crf.path_score(emissions, tags, mask)
paths = crf.viterbi(emissions, mask)
for b, n in enumerate(lengths):
    scores, bf_log_z = brute_force(crf, emissions[b, :n], n)
    bf_best = max(scores, key=scores.get)
    gold_seq = tuple(tags[b, :n].tolist())
    assert abs(log_z[b].item() - bf_log_z) < 1e-5, (b, log_z[b].item(), bf_log_z)
    assert abs(gold[b].item() - scores[gold_seq]) < 1e-5
    assert tuple(paths[b]) == bf_best, (b, paths[b], bf_best)
    # the same sequence alone, with no padding
    alone_mask = torch.ones(1, n, dtype=torch.bool, device=DEVICE)
    alone = emissions[b:b + 1, :n]
    assert abs(crf.log_partition(alone, alone_mask).item() - log_z[b].item()) < 1e-5
    assert crf.viterbi(alone, alone_mask)[0] == paths[b]
    print(f"seq {b} (len {n}, {K ** n:3d} sequences): logZ forward {log_z[b].item():9.5f} | brute force {bf_log_z:9.5f} | "
          f"Viterbi {paths[b]} == brute-force best {list(bf_best)}")
nll = crf.nll(emissions, tags, mask)
assert torch.all(nll >= -1e-5), "NLL must be non-negative"
print("CRF forward algorithm, path score and Viterbi agree with brute force; padding is handled correctly.")
''')

md(r'''
### 7.3 Overfitting sanity check

Training on 64 training utterances for 300 full-batch steps should fit them almost perfectly. If it does, the loss,
gradients and decoding are connected correctly. This is run for the CRF and softmax variants, using the real
architecture and dropout, and scored in eval mode with the shared evaluator. These 64 utterances are only used for
this check; the real training in Section 8 starts from a fresh model.
''')

code(r'''
def overfit_check(cfg, n_examples=64, steps=300, lr=1e-3):
    set_seed(SEED)
    model = build_model(cfg, emb_matrix)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    subset = encoded["train"][:n_examples]
    batch = to_device(collate(subset))
    for step in range(1, steps + 1):
        model.train()
        intent_logits, emissions = model(batch["words"], batch["chars"], batch["mask"])
        loss = model.losses(intent_logits, emissions, batch).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step in (1, 50, 100, 200, steps):
            s = evaluate_model(model, subset)
            print(f"  step {step:3d} | train-mode loss {loss.item():8.4f} | eval loss {s['loss']:7.4f} | {format_scores(s)}")
    return s

for name, overrides in [("full (char + CRF)", {}), ("no CRF (softmax)", {"use_crf": False})]:
    print(name)
    s = overfit_check({**MODEL_CONFIG, **overrides})
    assert s["intent_acc"] == 100.0 and s["slot_f1"] >= 99.0, s
print("Both variants fit the 64 examples.")
''')

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "main.ipynb"
nbformat.write(nb, out_path)
print(f"wrote {len(cells)} cells to {out_path}")
