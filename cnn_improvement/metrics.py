"""Exact MASSIVE intent/slot evaluation, independent of model internals.

Slots are per-utterance sets of (slot type, original span text), as specified
in the assignment. They are not token-level BIO F1 and are not matched by
character position. Character offsets are used only to preserve span text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SLOT_PATTERN = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")
TOKEN_PATTERN = re.compile(r"\S+")


def token_offsets(utterance: str) -> list[tuple[int, int]]:
    """Whitespace-token offsets in the unmodified input string."""
    return [(match.start(), match.end()) for match in TOKEN_PATTERN.finditer(utterance)]


def parse_annotated_utterance(
    annotated: str, utterance: str | None = None
) -> dict[str, Any]:
    """Remove markup while retaining independently parsed gold spans.

    Whitespace surrounding the type and value inside ``[type : value]`` is
    annotation syntax. Internal span whitespace and all outside text are
    preserved. Supplying ``utterance`` asserts exact reconstruction.
    """
    parts: list[str] = []
    spans: list[tuple[str, int, int]] = []
    position = 0
    output_length = 0
    for match in SLOT_PATTERN.finditer(annotated):
        outside = annotated[position:match.start()]
        if "[" in outside or "]" in outside:
            raise ValueError("Unparsed slot annotation bracket")
        parts.append(outside)
        output_length += len(outside)
        slot_type, value = match.group(1).strip(), match.group(2)
        if not slot_type or not value or not value.split():
            raise ValueError("Empty slot type or slot value")
        start = output_length
        parts.append(value)
        output_length += len(value)
        spans.append((slot_type, start, output_length))
        position = match.end()
    trailing = annotated[position:]
    if "[" in trailing or "]" in trailing:
        raise ValueError("Unparsed slot annotation bracket")
    parts.append(trailing)
    reconstructed = "".join(parts)
    if utterance is not None and reconstructed != utterance:
        raise ValueError(
            "Removing slot markup does not reproduce utt exactly: "
            f"{reconstructed!r} != {utterance!r}"
        )
    return {
        "utt": reconstructed,
        "spans": spans,
        "gold_pairs": {(kind, reconstructed[start:end]) for kind, start, end in spans},
    }


def gold_bio_tags(utterance: str, annotated: str) -> list[str]:
    """Align gold annotation spans to whitespace tokens, without dropping data."""
    parsed = parse_annotated_utterance(annotated, utterance)
    offsets = token_offsets(utterance)
    if not offsets:
        raise ValueError("Empty utterance cannot be used by this sequence model")
    by_start = {start: i for i, (start, _) in enumerate(offsets)}
    by_end = {end: i for i, (_, end) in enumerate(offsets)}
    tags = ["O"] * len(offsets)
    for slot_type, start, end in parsed["spans"]:
        if start not in by_start or end not in by_end:
            raise ValueError(f"Slot {slot_type!r} does not align with whitespace-token boundaries")
        first, last = by_start[start], by_end[end]
        if last < first or any(tag != "O" for tag in tags[first:last + 1]):
            raise ValueError("Overlapping or empty slot annotation")
        tags[first] = f"B-{slot_type}"
        for i in range(first + 1, last + 1):
            tags[i] = f"I-{slot_type}"
    return tags


def tags_to_pairs(utterance: str, tags: Sequence[str]) -> set[tuple[str, str]]:
    """Decode BIO tags into exact original-text pairs.

    A B-tag always opens a new span. An orphan I-tag opens a new span of its
    type, matching the baseline branch's documented recovery convention.
    Padding must be removed by the caller; length mismatches are errors.
    """
    offsets = token_offsets(utterance)
    if len(tags) != len(offsets):
        raise ValueError(f"Tag/token count mismatch: {len(tags)} != {len(offsets)}")
    result: set[tuple[str, str]] = set()
    active_type: str | None = None
    active_start = active_end = 0
    for (start, end), tag in zip(offsets, tags):
        if tag == "O":
            prefix, slot_type = "O", None
        else:
            if not isinstance(tag, str) or "-" not in tag:
                raise ValueError(f"Invalid BIO tag: {tag!r}")
            prefix, slot_type = tag.split("-", 1)
            if prefix not in {"B", "I"} or not slot_type:
                raise ValueError(f"Invalid BIO tag: {tag!r}")
        if prefix == "I" and slot_type == active_type:
            active_end = end
            continue
        if active_type is not None:
            result.add((active_type, utterance[active_start:active_end]))
        if prefix in {"B", "I"}:
            active_type, active_start, active_end = slot_type, start, end
        else:
            active_type = None
    if active_type is not None:
        result.add((active_type, utterance[active_start:active_end]))
    return result


def score_pair_predictions(
    gold_intents: Sequence[str | int],
    predicted_intents: Sequence[str | int],
    gold_pairs: Sequence[Iterable[tuple[str, str]]],
    predicted_pairs: Sequence[Iterable[tuple[str, str]]],
) -> dict[str, int | float]:
    """Pool exact slot TP/FP/FN over utterances and return percentage metrics."""
    n = len(gold_intents)
    if n == 0 or not (n == len(predicted_intents) == len(gold_pairs) == len(predicted_pairs)):
        raise ValueError("Nonempty gold and predicted collections must have equal lengths")
    intent_correct = sum(gold == pred for gold, pred in zip(gold_intents, predicted_intents))
    tp = fp = fn = 0
    for gold, predicted in zip(gold_pairs, predicted_pairs):
        gold_set, predicted_set = set(gold), set(predicted)
        tp += len(gold_set & predicted_set)
        fp += len(predicted_set - gold_set)
        fn += len(gold_set - predicted_set)
    intent_acc = 100.0 * intent_correct / n
    denominator = 2 * tp + fp + fn
    slot_f1 = 100.0 * 2 * tp / denominator if denominator else 100.0
    return {
        "n": n,
        "intent_correct": int(intent_correct),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "intent_acc": intent_acc,
        "slot_f1": slot_f1,
        "combined": (intent_acc + slot_f1) / 2.0,
    }


def score_predictions(
    records: Sequence[Mapping[str, Any]],
    intent_predictions: Sequence[int | str],
    tag_predictions: Sequence[Sequence[int | str]],
    vocab: Mapping[str, Any] | None = None,
) -> dict[str, int | float]:
    """Score raw records against predicted intent IDs and variable-length BIO IDs."""
    if not (len(records) == len(intent_predictions) == len(tag_predictions)):
        raise ValueError("Prediction and record counts differ")
    if vocab is None and any(not isinstance(value, str) for value in intent_predictions):
        raise ValueError("A vocabulary is required to decode numeric intent predictions")
    intents = [p if isinstance(p, str) else vocab["intents"][int(p)] for p in intent_predictions]
    predicted_pairs = []
    for record, tags in zip(records, tag_predictions):
        if vocab is None and any(not isinstance(tag, str) for tag in tags):
            raise ValueError("A vocabulary is required to decode numeric tag predictions")
        names = [tag if isinstance(tag, str) else vocab["tags"][int(tag)] for tag in tags]
        predicted_pairs.append(tags_to_pairs(record["utt"], names))
    return score_pair_predictions(
        [record["intent"] for record in records],
        intents,
        [parse_annotated_utterance(record["annot_utt"], record["utt"])["gold_pairs"] for record in records],
        predicted_pairs,
    )
