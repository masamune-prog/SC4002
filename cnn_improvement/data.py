"""Training-only vocabularies and lossless MASSIVE 1.0 preprocessing."""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .metrics import gold_bio_tags, parse_annotated_utterance, tags_to_pairs
except ImportError:  # Also support running train.py directly from this directory.
    from metrics import gold_bio_tags, parse_annotated_utterance, tags_to_pairs


PAD = "<<PAD>>"
UNK = "<<UNK>>"
PAD_ID = 0
UNK_ID = 1
SLOT_PAD_ID = -100
EXPECTED_COUNTS = {"train": 11514, "dev": 2033, "test": 2974}


def load_splits(path: str | Path, verify_counts: bool = True) -> dict[str, list[dict[str, Any]]]:
    """Read the official partition field; never re-split, discard, or add rows."""
    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in EXPECTED_COUNTS}
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"utt", "annot_utt", "intent", "partition", "id"} - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
            if record.get("locale", "en-US") != "en-US":
                raise ValueError("Only the official en-US locale is permitted")
            partition = record["partition"]
            if partition not in splits:
                raise ValueError(f"Unexpected partition {partition!r}")
            identifier = str(record["id"])
            if identifier in seen_ids:
                raise ValueError(f"Duplicate example id: {identifier}")
            seen_ids.add(identifier)
            splits[partition].append(record)
    counts = {name: len(rows) for name, rows in splits.items()}
    if verify_counts and counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected official MASSIVE 1.0 split counts {EXPECTED_COUNTS}; got {counts}")
    return splits


def build_vocab(train_records: Sequence[Mapping[str, Any]], verify_labels: bool = True) -> dict[str, Any]:
    """Build every inventory and word frequency from training records only.

    The plain lists/dicts/integers can be stored in JSON or a PyTorch checkpoint.
    Every vocabulary has stable sorted order; special word/character rows are
    PAD=0 and UNK=1. No external embedding words enter the input vocabulary.
    """
    if not train_records:
        raise ValueError("Cannot build vocabularies from an empty training set")
    if any(record.get("partition", "train") != "train" for record in train_records):
        raise ValueError("build_vocab accepts training records only")
    counts = Counter(word for record in train_records for word in record["utt"].split())
    if PAD in counts or UNK in counts:
        raise ValueError("Input text collides with a reserved vocabulary symbol")
    words = [PAD, UNK] + sorted(counts)
    chars = [PAD, UNK] + sorted({char for word in counts for char in word})
    intents = sorted({record["intent"] for record in train_records})
    slot_types = sorted({
        slot_type
        for record in train_records
        for slot_type, _, _ in parse_annotated_utterance(record["annot_utt"], record["utt"])["spans"]
    })
    tags = ["O"] + [f"{prefix}-{kind}" for kind in slot_types for prefix in ("B", "I")]
    if verify_labels and (len(intents), len(slot_types), len(tags)) != (60, 55, 111):
        raise ValueError(
            "Expected training label inventory of 60 intents, 55 slot types, 111 BIO tags; "
            f"got {(len(intents), len(slot_types), len(tags))}"
        )
    return {
        "words": words,
        "chars": chars,
        "intents": intents,
        "slot_types": slot_types,
        "tags": tags,
        "word_to_id": {word: i for i, word in enumerate(words)},
        "char_to_id": {char: i for i, char in enumerate(chars)},
        "intent_to_id": {intent: i for i, intent in enumerate(intents)},
        "tag_to_id": {tag: i for i, tag in enumerate(tags)},
        "word_counts": [0, 0] + [int(counts[word]) for word in words[2:]],
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
    }


def encode_record(record: Mapping[str, Any], vocab: Mapping[str, Any]) -> dict[str, Any]:
    """Encode inputs from utt only, with supervision parsed separately."""
    utterance = record["utt"]
    tokens = utterance.split()
    tags = gold_bio_tags(utterance, record["annot_utt"])
    parsed = parse_annotated_utterance(record["annot_utt"], utterance)
    if tags_to_pairs(utterance, tags) != parsed["gold_pairs"]:
        raise ValueError(f"Gold tag/span round trip failed for record {record.get('id')}")
    try:
        tag_ids = [vocab["tag_to_id"][tag] for tag in tags]
        intent_id = vocab["intent_to_id"][record["intent"]]
    except KeyError as exc:
        raise ValueError(f"Label absent from training vocabulary: {exc}") from exc
    return {
        "id": str(record.get("id", "")),
        "utt": utterance,
        "tokens": tokens,
        "word_ids": [vocab["word_to_id"].get(word, UNK_ID) for word in tokens],
        "char_ids": [[vocab["char_to_id"].get(char, UNK_ID) for char in word] for word in tokens],
        "intent_id": intent_id,
        "slot_ids": tag_ids,
        "intent": record["intent"],
        "gold_pairs": parsed["gold_pairs"],
    }


class MassiveDataset(Dataset):
    """Pre-encode records without changing their order or truncating sequences."""

    def __init__(self, records: Sequence[Mapping[str, Any]], vocab: Mapping[str, Any]):
        self.records = list(records)
        self.examples = [encode_record(record, vocab) for record in records]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_batch(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad words/characters/tags without truncation; include untouched metadata."""
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(examples)
    max_tokens = max(len(example["word_ids"]) for example in examples)
    max_chars = max(len(chars) for example in examples for chars in example["char_ids"])
    word_ids = torch.full((batch_size, max_tokens), PAD_ID, dtype=torch.long)
    char_ids = torch.full((batch_size, max_tokens, max_chars), PAD_ID, dtype=torch.long)
    slot_ids = torch.full((batch_size, max_tokens), SLOT_PAD_ID, dtype=torch.long)
    lengths = torch.tensor([len(example["word_ids"]) for example in examples], dtype=torch.long)
    for i, example in enumerate(examples):
        length = len(example["word_ids"])
        word_ids[i, :length] = torch.tensor(example["word_ids"], dtype=torch.long)
        slot_ids[i, :length] = torch.tensor(example["slot_ids"], dtype=torch.long)
        for j, chars in enumerate(example["char_ids"]):
            char_ids[i, j, :len(chars)] = torch.tensor(chars, dtype=torch.long)
    return {
        "word_ids": word_ids,
        "char_ids": char_ids,
        "lengths": lengths,
        "mask": torch.arange(max_tokens).unsqueeze(0) < lengths.unsqueeze(1),
        "intent_ids": torch.tensor([example["intent_id"] for example in examples], dtype=torch.long),
        "slot_ids": slot_ids,
        "ids": [example["id"] for example in examples],
        "utts": [example["utt"] for example in examples],
        "tokens": [example["tokens"] for example in examples],
        "gold_intents": [example["intent"] for example in examples],
        "gold_pairs": [example["gold_pairs"] for example in examples],
    }


@contextmanager
def _embedding_lines(path: str | Path, dimension: int):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if name.endswith(f".{dimension}d.txt")]
            if len(matches) != 1:
                raise ValueError(f"Expected one {dimension}d embedding text file in archive, got {matches}")
            with archive.open(matches[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8") as stream:
                yield stream
    else:
        with path.open(encoding="utf-8") as stream:
            yield stream


def load_embedding_matrix(
    path: str | Path,
    vocab: Mapping[str, Any],
    dim: int,
    seed: int = 42,
    strip_periods_fallback: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load GloVe rows for the training vocabulary only, from text or a ZIP.

    The optional exact-then-period-stripped lookup matches the inspected Max
    baseline. It affects lookup only: model tokens and output strings stay
    unchanged. Unknown rows use a seeded normal distribution whose scale is
    measured from the matched training-word vectors, and PAD stays zero.
    """
    words = vocab["words"]
    train_words = words[2:]
    wanted = set(train_words)
    if strip_periods_fallback:
        wanted.update(word.replace(".", "") for word in train_words)
    wanted.discard("")
    found: dict[str, np.ndarray] = {}
    with _embedding_lines(path, dim) as stream:
        for line in stream:
            word, separator, rest = line.rstrip("\r\n").partition(" ")
            if separator and word in wanted:
                vector = np.fromstring(rest, sep=" ", dtype=np.float32)
                if vector.shape != (dim,) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid embedding row for {word!r}: expected {dim} finite values")
                found[word] = vector
    if not found:
        raise ValueError("No training vocabulary words matched the embedding file")
    standard_deviation = float(np.stack(list(found.values())).std())
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Matched word vectors have invalid variance")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, standard_deviation, (len(words), dim)).astype(np.float32)
    matrix[PAD_ID] = 0
    exact = fallback = matched_tokens = 0
    counts = vocab["word_counts"]
    for index, word in enumerate(words[2:], start=2):
        if word in found:
            matrix[index] = found[word]
            exact += 1
            matched_tokens += counts[index]
        elif strip_periods_fallback and word.replace(".", "") in found:
            matrix[index] = found[word.replace(".", "")]
            fallback += 1
            matched_tokens += counts[index]
    total_tokens = sum(counts)
    coverage = {
        "dimension": dim,
        "vocabulary_rows": len(words),
        "train_types": len(train_words),
        "exact_types": exact,
        "fallback_types": fallback,
        "matched_types": exact + fallback,
        "unmatched_types": len(train_words) - exact - fallback,
        "type_coverage_pct": 100.0 * (exact + fallback) / len(train_words),
        "train_tokens": total_tokens,
        "matched_tokens": matched_tokens,
        "token_coverage_pct": 100.0 * matched_tokens / total_tokens,
        "random_init_std": standard_deviation,
        "source": str(Path(path)),
        "external_vocabulary_rows_added": 0,
    }
    return matrix, coverage
