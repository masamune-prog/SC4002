"""Evaluate a frozen BPE checkpoint on MASSIVE en-US test.csv.

The model receives only IDs derived from ``utt``. Gold intent and slot labels
are read solely to compute the assignment metrics. This script never trains.
"""

import argparse
import csv
import json
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from main import EncodedDataset, collate, evaluate, read_split
from main_transformer_crf import collate_crf, evaluate_crf
from sc4002.eval import parse_slots
from sc4002.models import JointTransformer, JointTransformerCRF
from sc4002.tokenizers import BpeTokenizer


def save_predictions(path, examples, pred_intents, pred_slots) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "id", "utt", "gold_intent", "pred_intent", "gold_slots",
            "pred_slots", "intent_correct", "slots_exact", "exact_match",
        ])
        writer.writeheader()
        for example, intent, slots in zip(examples, pred_intents, pred_slots):
            gold_slots = parse_slots(example.annot_utt)
            intent_correct = example.intent == intent
            slots_exact = Counter(gold_slots) == Counter(slots)
            writer.writerow({
                "id": example.record_id,
                "utt": example.utt,
                "gold_intent": example.intent,
                "pred_intent": intent,
                "gold_slots": json.dumps(gold_slots, ensure_ascii=False),
                "pred_slots": json.dumps(slots, ensure_ascii=False),
                "intent_correct": int(intent_correct),
                "slots_exact": int(slots_exact),
                "exact_match": int(intent_correct and slots_exact),
            })


def save_results(path, checkpoint_path, saved, model, examples, loss, scores) -> None:
    row = {
        "checkpoint": str(checkpoint_path),
        "model_type": saved.get("model_type", "joint_transformer"),
        "selected_epoch": saved["epoch"],
        "split": "test",
        "examples": len(examples),
        "trainable_parameters": model.parameter_count(),
        "loss": f"{loss:.4f}",
        "intent_accuracy_pct": f"{100 * scores['intent_accuracy']:.2f}",
        "slot_precision_pct": f"{100 * scores['slot_precision']:.2f}",
        "slot_recall_pct": f"{100 * scores['slot_recall']:.2f}",
        "slot_micro_f1_pct": f"{100 * scores['slot_f1']:.2f}",
        "exact_match_pct": f"{100 * scores['exact_match']:.2f}",
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def evaluate_checkpoint(
    checkpoint_path: Path, test_csv: Path, batch_size: int,
    output_dir: Path = Path("outputs"),
) -> dict[str, float]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint_path}. Train a model first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if saved.get("tokenizer_type") != "bpe" or "tokenizer_json" not in saved:
        raise ValueError("Checkpoint does not contain a fitted BPE tokenizer")
    tokenizer = BpeTokenizer.from_json(saved["tokenizer_json"])
    model_type = saved.get("model_type", "joint_transformer")
    if model_type == "transformer_crf":
        model_config = dict(saved["model_config"])
        if "num_layers" not in model_config:
            # Checkpoints made before the deeper CRF stored only shared settings.
            model_config.update(
                d_model=256, nhead=4, num_layers=3,
                dim_feedforward=512, norm_first=False,
            )
        model = JointTransformerCRF(**model_config).to(device)
        collate_fn = partial(collate_crf, pad_id=tokenizer.pad_id)
    elif model_type == "joint_transformer":
        model = JointTransformer(**saved["model_config"]).to(device)
        collate_fn = partial(collate, pad_id=tokenizer.pad_id)
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type!r}")
    model.load_state_dict(saved["model_state"])

    test = read_split(test_csv, "test", 2_974)
    loader = DataLoader(
        EncodedDataset(test, tokenizer, saved["intents"], saved["tags"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    if model_type == "transformer_crf":
        loss, scores, pred_intents, pred_slots = evaluate_crf(
            model, loader, test, saved["intents"], saved["tags"], device,
            saved.get("slot_weight", 1.0), return_predictions=True,
        )
    else:
        loss, scores, pred_intents, pred_slots = evaluate(
            model, loader, test, tokenizer, saved["intents"], saved["tags"], device,
            return_predictions=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "test_predictions.csv"
    results_path = output_dir / "test_results.csv"
    save_predictions(predictions_path, test, pred_intents, pred_slots)
    save_results(results_path, checkpoint_path, saved, model, test, loss, scores)
    print(f"checkpoint: {checkpoint_path}")
    print(f"selected epoch: {saved['epoch']}")
    print(f"parameters: {model.parameter_count():,} / 20,000,000")
    print(f"test loss: {loss:.4f}")
    print(f"intent accuracy: {100 * scores['intent_accuracy']:.2f}%")
    print(f"slot micro F1: {100 * scores['slot_f1']:.2f}%")
    print(f"predictions: {predictions_path}")
    print(f"results: {results_path}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/bpe_transformer/best_bpe_transformer.pt"),
    )
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.test_csv, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
