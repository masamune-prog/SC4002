"""Shared scoring rules: intent accuracy and exact-match slot micro F1 (both in %)."""

from __future__ import annotations

from collections import Counter


def tags_to_spans(words: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """BIO tags -> (type, span text). An I-x that does not continue an x span starts a new span."""
    spans, cur_type, cur_words = [], None, []
    for w, t in zip(words, tags):
        if t == "O":
            prefix, typ = "O", None
        else:
            prefix, typ = t.split("-", 1)
        if prefix == "B" or (prefix == "I" and typ != cur_type) or prefix == "O":
            if cur_type is not None:
                spans.append((cur_type, " ".join(cur_words)))
            cur_type, cur_words = (typ, [w]) if prefix != "O" else (None, [])
        else:
            cur_words.append(w)
    if cur_type is not None:
        spans.append((cur_type, " ".join(cur_words)))
    return spans


def intent_accuracy(gold: list[str], pred: list[str]) -> float:
    assert len(gold) == len(pred)
    return 100.0 * sum(g == p for g, p in zip(gold, pred)) / len(gold)


def slot_counts(gold: list[list[tuple[str, str]]], pred: list[list[tuple[str, str]]]):
    """TP/FP/FN pooled over utterances. Per utterance, pairs are compared as multisets,
    so a (type, span) that occurs twice must be predicted twice."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred, strict=True):
        g, p = Counter(g), Counter(p)
        hit = sum((g & p).values())
        tp += hit
        fp += sum(p.values()) - hit
        fn += sum(g.values()) - hit
    return tp, fp, fn


def slot_micro_f1(gold, pred) -> float:
    tp, fp, fn = slot_counts(gold, pred)
    denom = 2 * tp + fp + fn
    return 100.0 * 2 * tp / denom if denom else 100.0


def score(gold_intents, pred_intents, gold_slots, pred_slots) -> dict:
    tp, fp, fn = slot_counts(gold_slots, pred_slots)
    acc = intent_accuracy(gold_intents, pred_intents)
    f1 = slot_micro_f1(gold_slots, pred_slots)
    return dict(intent_acc=acc, slot_f1=f1, combined=(acc + f1) / 2, tp=tp, fp=fp, fn=fn,
                n=len(gold_intents))


def fmt(s: dict) -> str:
    return f"intent acc {s['intent_acc']:.2f}  slot F1 {s['slot_f1']:.2f}  combined {s['combined']:.2f}"
