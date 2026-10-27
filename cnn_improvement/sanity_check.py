"""Overfit 64 fixed training examples to check gradients and task wiring.

This diagnostic uses random word vectors and the real architecture, dropout,
loss, and evaluator. Its scores are training-subset checks, never project
validation/test results. It does not encode or evaluate dev or test records.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from .data import MassiveDataset, build_vocab, collate_batch
from .train import build_model, configure, environment, evaluate, json_write, move_batch


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "cnn192.json"))
    parser.add_argument("--data", default=str(ROOT / "data" / "en-US.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "diagnostics" / "sanity.json"))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = configure(20261026, args.threads)
    train_records = []
    with Path(args.data).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["partition"] == "train":
                train_records.append(row)
    assert len(train_records) == 11514
    vocab = build_vocab(train_records)
    indices = random.Random(20261026).sample(range(len(train_records)), 64)
    tiny = MassiveDataset([train_records[index] for index in indices], vocab)
    model = build_model(config["model"], vocab, device=device)
    batch = move_batch(collate_batch(tiny.examples), device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.003, weight_decay=config["training"]["weight_decay"],
    )
    initial = evaluate(model, tiny, vocab, device, batch_size=64, include_loss=True)
    print("Diagnostic only: 64 training examples, random embeddings, no dev/test evaluation.", flush=True)
    print(f"Before: intent {initial['intent_acc']:.2f}%, exact slot F1 {initial['slot_f1']:.2f}%", flush=True)
    history = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["word_ids"], batch["char_ids"], batch["lengths"])
        loss, components = model.compute_loss(*output, batch["intent_ids"], batch["slot_ids"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite sanity loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            metrics = evaluate(model, tiny, vocab, device, batch_size=64, include_loss=True)
            history.append({"step": step, "training_loss": float(loss.detach()),
                            "eval_on_training_subset": metrics})
            print(f"Step {step:03d}: train loss {float(loss.detach()):.4f}, "
                  f"subset intent {metrics['intent_acc']:.2f}%, exact slot F1 {metrics['slot_f1']:.2f}%",
                  flush=True)
            if metrics["intent_acc"] >= 99.0 and metrics["slot_f1"] >= 99.0:
                break
    final = evaluate(model, tiny, vocab, device, batch_size=64, include_loss=True)
    passed = final["intent_acc"] >= 95.0 and final["slot_f1"] >= 95.0
    results = {
        "purpose": "Training-subset overfit diagnostic, not a validation or test result",
        "seed": 20261026,
        "n_training_examples": 64,
        "training_ids": [row["id"] for row in tiny.records],
        "word_embedding_initialization": "random, sanity check only",
        "config": config["model"],
        "optimizer": {"name": "AdamW", "learning_rate": 0.003,
                      "weight_decay": config["training"]["weight_decay"]},
        "steps": step,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment(device),
        "initial": initial,
        "final": final,
        "history": history,
        "pass_threshold_percent": 95.0,
        "passed": passed,
        "test_evaluated": False,
        "validation_evaluated": False,
    }
    json_write(args.output, results)
    print(f"Sanity {'PASSED' if passed else 'FAILED'}: "
          f"intent {final['intent_acc']:.2f}%, exact slot F1 {final['slot_f1']:.2f}%", flush=True)
    if not passed:
        raise AssertionError("Tiny-training-subset overfit check did not reach both 95% thresholds")


if __name__ == "__main__":
    main()
