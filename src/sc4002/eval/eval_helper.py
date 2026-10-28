import re
from collections import Counter
from collections.abc import Sequence

Slot = tuple[str, str]
SlotPrediction = str | Sequence[Slot]

# Matches MASSIVE slot annotations such as "[time : nine am]".
# The slot type cannot contain ":" but the value can (e.g. "[time : 5:30 pm]").
SLOT_PATTERN = re.compile(r"\[\s*([^:\[\]]+?)\s*:\s*([^\[\]]+?)\s*\]")


def parse_slots(annot_utt: str) -> list[tuple[str, str]]:
    """Extract (slot_type, value) pairs from an annotated utterance."""
    return [
        (slot_type.strip(), " ".join(value.split()))
        for slot_type, value in SLOT_PATTERN.findall(annot_utt)
    ]


def _slots(prediction: SlotPrediction) -> list[Slot]:
    """Accept an annotated string or already decoded span pairs."""
    if isinstance(prediction, str):
        return parse_slots(prediction)
    return [(slot_type, " ".join(value.split())) for slot_type, value in prediction]


class EvalMetric:
    """Evaluate model performance on unseen test data."""

    def __init__(self, true_intents: Sequence[str], true_annots: Sequence[str]):
        if len(true_intents) != len(true_annots):
            raise ValueError("true_intents and true_annots must have the same length")
        self.true_intents = list(true_intents)
        self.true_slots = [parse_slots(a) for a in true_annots]

    def _check_length(self, preds: Sequence) -> None:
        if len(preds) != len(self.true_intents):
            raise ValueError(
                f"Expected {len(self.true_intents)} predictions, got {len(preds)}"
            )

    def intent_accuracy(self, pred_intents: Sequence[str]) -> float:
        """Fraction of utterances whose predicted intent matches the gold intent."""
        self._check_length(pred_intents)
        if not self.true_intents:
            return 0.0
        correct = sum(t == p for t, p in zip(self.true_intents, pred_intents))
        return correct / len(self.true_intents)

    def slot_scores(self, pred_annots: Sequence[SlotPrediction]) -> dict[str, float]:
        """Micro precision/recall/F1 over (slot_type, value) spans.

        If there are no gold and no predicted slots at all, every score is 1.0.
        """
        self._check_length(pred_annots)
        tp = n_true = n_pred = 0
        for gold, annot in zip(self.true_slots, pred_annots):
            pred = _slots(annot)
            tp += sum((Counter(gold) & Counter(pred)).values())
            n_true += len(gold)
            n_pred += len(pred)

        if n_true == 0 and n_pred == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_true if n_true else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    def exact_match(
        self, pred_intents: Sequence[str], pred_annots: Sequence[SlotPrediction]
    ) -> float:
        """Fraction of utterances with the correct intent and exactly the gold slots."""
        self._check_length(pred_intents)
        self._check_length(pred_annots)
        if not self.true_intents:
            return 0.0
        correct = sum(
            t_int == p_int and Counter(t_slots) == Counter(_slots(p_annot))
            for t_int, p_int, t_slots, p_annot in zip(
                self.true_intents, pred_intents, self.true_slots, pred_annots
            )
        )
        return correct / len(self.true_intents)

    def evaluate(
        self, pred_intents: Sequence[str], pred_annots: Sequence[SlotPrediction]
    ) -> dict[str, float]:
        """Compute all metrics at once."""
        slots = self.slot_scores(pred_annots)
        return {
            "intent_accuracy": self.intent_accuracy(pred_intents),
            "slot_precision": slots["precision"],
            "slot_recall": slots["recall"],
            "slot_f1": slots["f1"],
            "exact_match": self.exact_match(pred_intents, pred_annots),
        }

    def evaluate_percent(
        self, pred_intents: Sequence[str], pred_slots: Sequence[SlotPrediction]
    ) -> dict[str, float]:
        """Return assignment-ready percentages rounded to two decimal places."""
        return {
            key: round(100 * value, 2)
            for key, value in self.evaluate(pred_intents, pred_slots).items()
        }
