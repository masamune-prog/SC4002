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
''')

code(r'''
import os
# cuBLAS needs this workspace setting for deterministic kernels; it must be set before torch is imported.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "main.ipynb"
nbformat.write(nb, out_path)
print(f"wrote {len(cells)} cells to {out_path}")
