"""Data loading, annotation parsing, tokenisation and batching for MASSIVE en-US."""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLIT_SIZES = {"train": 11514, "dev": 2033, "test": 2974}

SLOT_PATTERN = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")
PAD, UNK = "<<pad>>", "<<unk>>"  # double brackets: a literal "<empty>" token occurs in the data


@dataclass
class Example:
    id: str
    utt: str
    words: list[str]
    intent: str
    tags: list[str]  # word-level BIO tags
    gold: list[tuple[str, str]]  # (slot type, span text) read directly from annot_utt


def parse_annot(annot_utt: str) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Turn an annotated utterance into whitespace words, BIO tags and gold (type, span) pairs."""
    words, tags, gold = [], [], []
    pos = 0
    for m in SLOT_PATTERN.finditer(annot_utt):
        outside = annot_utt[pos : m.start()].split()
        words += outside
        tags += ["O"] * len(outside)
        slot, span = m.group(1).strip(), m.group(2).split()
        words += span
        tags += [f"B-{slot}"] + [f"I-{slot}"] * (len(span) - 1)
        gold.append((slot, " ".join(span)))
        pos = m.end()
    outside = annot_utt[pos:].split()
    words += outside
    tags += ["O"] * len(outside)
    return words, tags, gold


def load_split(name: str) -> list[Example]:
    with open(DATA_DIR / f"{name}.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        assert r["partition"] == name and r["locale"] == "en-US"
        words, tags, gold = parse_annot(r["annot_utt"])
        assert words == r["utt"].split(), (r["id"], r["utt"], r["annot_utt"])
        out.append(Example(r["id"], r["utt"], words, r["intent"], tags, gold))
    return out


def load_splits() -> dict[str, list[Example]]:
    splits = {name: load_split(name) for name in SPLIT_SIZES}
    for name, size in SPLIT_SIZES.items():
        assert len(splits[name]) == size, (name, len(splits[name]))
    return splits


# ---------------------------------------------------------------- tokenisation
# Every tokeniser maps a whitespace word to one or more "units". The encoder runs over
# units; slot tags stay word-level and are read from each word's first unit, so scoring
# is identical whichever tokeniser is used.

_CLITIC = re.compile(r"^(.+?)(n't|'s|'re|'ll|'d|'ve|'m)$")


def clitic_units(word: str) -> list[str]:
    """what's -> what 's, don't -> do n't, p. -> p ."""
    m = _CLITIC.match(word)
    if m and m.group(1) not in {"'", ""}:
        return [m.group(1), m.group(2)]
    if len(word) > 1 and word[-1] in ".,?!" and word[:-1].isalnum():
        return [word[:-1], word[-1]]
    return [word]


class Tokeniser:
    """kind: 'word' | 'clitic' | 'bpe'. BPE is trained on train utterances only.

    The `tokenizers` BPE trainer is not deterministic across processes (vocabulary ids and merge
    ties change from run to run), so with save_dir set a trained BPE is saved to
    save_dir/bpe_<vocab>.json once and loaded by every later run."""

    def __init__(self, kind: str, train: list[Example] | None = None, bpe_vocab: int = 4000,
                 save_dir: Path | None = None):
        self.kind = kind
        self.bpe = None
        if kind == "bpe":
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers

            path = save_dir / f"bpe_{bpe_vocab}.json" if save_dir else None
            if path and path.exists():
                self.bpe = Tokenizer.from_file(str(path))
                self._cache: dict[str, list[str]] = {}
                return
            tok = Tokenizer(models.BPE(unk_token=UNK, continuing_subword_prefix="##"))
            tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
            trainer = trainers.BpeTrainer(
                vocab_size=bpe_vocab, special_tokens=[PAD, UNK], continuing_subword_prefix="##"
            )
            tok.train_from_iterator((ex.utt for ex in train), trainer=trainer)
            if path:
                tok.save(str(path))
            self.bpe = tok
        self._cache: dict[str, list[str]] = {}

    def word_units(self, word: str) -> list[str]:
        if word not in self._cache:
            if self.kind == "word":
                units = [word]
            elif self.kind == "clitic":
                units = clitic_units(word)
            else:
                units = self.bpe.encode(word).tokens or [UNK]
            self._cache[word] = units
        return self._cache[word]

    def vector_keys(self, word: str, units: list[str]) -> list[str]:
        """Which string to look up in the pretrained vector table for each unit.
        Subword pieces have no pretrained vectors, so they share their parent word's."""
        return [word] * len(units) if self.kind == "bpe" else units


class Vocab:
    def __init__(self, items, specials=(PAD, UNK)):
        self.itos = list(specials) + [x for x in items if x not in specials]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, s):
        return self.stoi.get(s, self.stoi.get(UNK, 0))


def label_vocabs(train: list[Example]) -> tuple[Vocab, Vocab]:
    intents = Vocab(sorted({ex.intent for ex in train}), specials=())
    slot_types = sorted({t[2:] for ex in train for t in ex.tags if t != "O"})
    tags = Vocab(["O"] + [f"{p}-{s}" for s in slot_types for p in "BI"], specials=())
    assert len(intents) == 60 and len(slot_types) == 55 and len(tags) == 111
    return intents, tags


# ---------------------------------------------------------------- encoding


@dataclass
class Encoded:
    units: list[int]
    unit_counts: list[int]  # train frequency of each unit (for word dropout)
    vec_rows: list[list[int]]  # pretrained-table rows averaged for each unit ([] = none)
    chars: list[list[int]]
    word_first: list[int]  # index of each word's first unit
    intent: int
    tags: list[int]
    ex: Example = field(repr=False)


class Encoder:
    """Turns Examples into index tensors. All vocabularies are fit on train only."""

    def __init__(self, train: list[Example], tokeniser: Tokeniser, vectors=None, max_chars: int = 20):
        self.tok = tokeniser
        self.vectors = vectors  # sc4002.vectors.VectorTable or None
        self.max_chars = max_chars
        counts = Counter(u for ex in train for w in ex.words for u in tokeniser.word_units(w))
        self.unit_counts = counts
        self.units = Vocab([u for u, _ in counts.most_common()])
        self.chars = Vocab(sorted({c for u in counts for c in u}))
        self.intents, self.tags = label_vocabs(train)

    def encode(self, ex: Example) -> Encoded:
        units, counts, rows, chars, first = [], [], [], [], []
        for w in ex.words:
            us = self.tok.word_units(w)
            first.append(len(units))
            for u, key in zip(us, self.tok.vector_keys(w, us)):
                units.append(self.units[u])
                counts.append(self.unit_counts.get(u, 0))
                rows.append(self.vectors.lookup(key) if self.vectors else [])
                chars.append([self.chars[c] for c in u[: self.max_chars]] or [self.chars[UNK]])
        return Encoded(
            units, counts, rows, chars, first,
            self.intents[ex.intent], [self.tags[t] for t in ex.tags], ex,
        )


def collate(batch: list[Encoded], word_dropout: float = 0.0, rng: np.random.Generator | None = None):
    """Pad a batch. With word_dropout=a, a train unit seen c times becomes UNK with prob a/(a+c)."""
    B = len(batch)
    U = max(len(e.units) for e in batch)
    W = max(len(e.word_first) for e in batch)
    P = max([len(r) for e in batch for r in e.vec_rows] + [1])
    C = max(len(c) for e in batch for c in e.chars)
    units = torch.zeros(B, U, dtype=torch.long)
    vec = torch.zeros(B, U, P, dtype=torch.long)  # row 0 of the table is a zero vector
    chars = torch.zeros(B, U, C, dtype=torch.long)
    first = torch.zeros(B, W, dtype=torch.long)
    tags = torch.zeros(B, W, dtype=torch.long)
    umask = torch.zeros(B, U, dtype=torch.bool)
    wmask = torch.zeros(B, W, dtype=torch.bool)
    for i, e in enumerate(batch):
        ids = list(e.units)
        if word_dropout > 0:
            p = word_dropout / (word_dropout + np.asarray(e.unit_counts, dtype=float))
            drop = rng.random(len(ids)) < p
            ids = [1 if d else u for u, d in zip(ids, drop)]  # 1 == UNK
        units[i, : len(ids)] = torch.tensor(ids)
        umask[i, : len(ids)] = True
        for j, (r, c) in enumerate(zip(e.vec_rows, e.chars)):
            if r:
                vec[i, j, : len(r)] = torch.tensor(r)
            chars[i, j, : len(c)] = torch.tensor(c)
        first[i, : len(e.word_first)] = torch.tensor(e.word_first)
        tags[i, : len(e.tags)] = torch.tensor(e.tags)
        wmask[i, : len(e.word_first)] = True
    intents = torch.tensor([e.intent for e in batch])
    return dict(units=units, vec=vec, chars=chars, first=first, umask=umask, wmask=wmask,
                intents=intents, tags=tags)


def batches(data: list[Encoded], batch_size: int, shuffle: bool, rng: np.random.Generator | None = None):
    idx = np.arange(len(data))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield [data[i] for i in idx[s : s + batch_size]]
