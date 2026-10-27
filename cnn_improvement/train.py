"""Reproducible training, validation-only selection, and frozen final evaluation.

Run from the project workspace with ``python -m cnn_improvement.train --help``.
No notebook or repository code is downloaded or executed by this program.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import MassiveDataset, build_vocab, collate_batch, load_embedding_matrix, load_splits
from .metrics import parse_annotated_utterance, score_predictions, tags_to_pairs
from .model import JointDilatedCNN

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "en-US.jsonl"
DEFAULT_GLOVE = ROOT / "embeddings" / "glove.6B.300d.txt"


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    return {p.name: sha256(p) for p in sorted(ROOT.glob("*.py"))}


def configure(seed=42, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # A notebook can call this entry point more than once.
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment(device):
    return {
        "python": sys.version,
        "executable": sys.executable,
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(device),
        "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }


def build_model(config, vocab, matrix=None, device="cpu"):
    model = JointDilatedCNN(
        config, len(vocab["words"]), len(vocab["chars"]), len(vocab["intents"]),
        len(vocab["tags"]), vocab["tags"], matrix,
    ).to(device)
    assert model.parameter_counts()["total"] <= 20_000_000
    return model


def move_batch(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def epoch_batches(examples, batch_size, seed, bucket_multiplier=20):
    """Shuffle all examples once, length-bucket locally, then shuffle batches."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    batches = []
    window = batch_size * bucket_multiplier
    for start in range(0, len(order), window):
        bucket = sorted(order[start:start + window], key=lambda i: len(examples[i]["word_ids"]))
        batches.extend(bucket[i:i + batch_size] for i in range(0, len(bucket), batch_size))
    rng.shuffle(batches)
    assert sorted(index for batch in batches for index in batch) == list(range(len(examples)))
    return batches


@torch.inference_mode()
def average_crf_decode(models, emissions, mask):
    """Product-of-experts Viterbi with mean emissions and transition scores."""
    constrained = [model.crf.constrained_scores() for model in models]
    transitions, start, end = [torch.stack([scores[i] for scores in constrained]).mean(0)
                               for i in range(3)]
    batch_size, time_steps, n_tags = emissions.shape
    score = start + emissions[:, 0]
    identity = torch.arange(n_tags, device=emissions.device).expand(batch_size, -1)
    backpointers = []
    for t in range(1, time_steps):
        best, index = (score.unsqueeze(2) + transitions).max(1)
        active = mask[:, t:t + 1]
        score = torch.where(active, best + emissions[:, t], score)
        backpointers.append(torch.where(active, index, identity))
    tag = (score + end).argmax(-1)
    path = [tag]
    for pointer in reversed(backpointers):
        tag = pointer.gather(1, tag[:, None]).squeeze(1)
        path.append(tag)
    return torch.stack(path[::-1], 1).masked_fill(~mask, 0)


@torch.inference_mode()
def evaluate(models, dataset, vocab, device, batch_size=128, include_loss=False,
             save_predictions=None):
    """All models receive utt-derived tensors only; labels are used to score."""
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        model.eval()
    intent_ids, tag_ids = [], []
    intent_loss_sum = slot_loss_sum = 0.0
    n_tokens = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for start in range(0, len(dataset), batch_size):
        examples = dataset.examples[start:start + batch_size]
        batch = move_batch(collate_batch(examples), device)
        outputs = [model(batch["word_ids"], batch["char_ids"], batch["lengths"])
                   for model in models]
        probabilities = torch.stack([output[0].softmax(-1) for output in outputs]).mean(0)
        emissions = torch.stack([output[1] for output in outputs]).mean(0)
        mask = outputs[0][2]
        predictions = (models[0].decode(emissions, mask) if len(models) == 1
                       else average_crf_decode(models, emissions, mask))
        intent_ids.extend(probabilities.argmax(-1).cpu().tolist())
        tag_ids.extend(row[:length] for row, length in
                       zip(predictions.cpu().tolist(), batch["lengths"].cpu().tolist()))
        if include_loss:
            if len(models) != 1:
                raise ValueError("Training objective reporting is defined for a single member")
            _, components = models[0].compute_loss(*outputs[0], batch["intent_ids"], batch["slot_ids"])
            intent_loss_sum += components["intent_loss"].item() * len(examples)
            token_count = int(mask.sum())
            slot_denominator = token_count if models[0].config["slot_loss_normalization"] == "token" else len(examples)
            slot_loss_sum += components["slot_loss"].item() * slot_denominator
            n_tokens += token_count
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    metrics = score_predictions(dataset.records, intent_ids, tag_ids, vocab)
    metrics["inference_seconds"] = inference_seconds
    metrics["utterances_per_second"] = len(dataset) / inference_seconds
    if include_loss:
        metrics["intent_loss"] = intent_loss_sum / len(dataset)
        denominator = n_tokens if models[0].config["slot_loss_normalization"] == "token" else len(dataset)
        metrics["slot_loss"] = slot_loss_sum / denominator
        metrics["loss"] = metrics["intent_loss"] + models[0].config["slot_weight"] * metrics["slot_loss"]
    if save_predictions:
        with Path(save_predictions).open("w", encoding="utf-8") as stream:
            for record, intent, tags in zip(dataset.records, intent_ids, tag_ids):
                names = [vocab["tags"][tag] for tag in tags]
                row = {"id": record["id"], "utt": record["utt"], "gold_intent": record["intent"],
                       "predicted_intent": vocab["intents"][intent],
                       "gold_slots": sorted(parse_annotated_utterance(record["annot_utt"], record["utt"])["gold_pairs"]),
                       "predicted_slots": sorted(tags_to_pairs(record["utt"], names)),
                       "predicted_tags": names}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics


def cached_embeddings(glove_path, vocab, dimension, seed):
    cache = ROOT / "cache"
    cache.mkdir(exist_ok=True)
    vocabulary_hash = hashlib.sha256(json.dumps(vocab["words"]).encode()).hexdigest()[:16]
    target = cache / f"glove_{dimension}_{vocabulary_hash}_seed{seed}.npz"
    metadata = target.with_suffix(".json")
    if target.exists() and metadata.exists():
        matrix = np.load(target, allow_pickle=False)["matrix"]
        coverage = json.loads(metadata.read_text(encoding="utf-8"))
    else:
        matrix, coverage = load_embedding_matrix(glove_path, vocab, dimension, seed=seed)
        np.savez_compressed(target, matrix=matrix)
        json_write(metadata, coverage)
    assert matrix.shape == (len(vocab["words"]), dimension)
    return matrix, coverage


def learning_rate_scale(step, total_steps, warmup_steps, minimum):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tc = config["training"]
    if args.seed is not None:
        tc["seed"] = args.seed
    if args.threads is not None:
        tc["threads"] = args.threads
    if args.epochs is not None:
        tc["epochs"] = args.epochs
    device = configure(tc["seed"], tc["threads"])
    run_dir = Path(args.run_dir)
    if (run_dir / "summary.json").exists() and not args.resume:
        raise ValueError("Completed run exists; use a new run directory to preserve evidence")
    run_dir.mkdir(parents=True, exist_ok=True)
    data = load_splits(args.data)
    vocab = build_vocab(data["train"])
    training = MassiveDataset(data["train"], vocab)
    validation = MassiveDataset(data["dev"], vocab)
    # Test text is not encoded or scored during any training/configuration selection.
    matrix, coverage = cached_embeddings(args.glove, vocab, config["model"]["emb_dim"], tc["seed"])
    model = build_model(config["model"], vocab, matrix, device)
    config["model"] = model.config
    counts = model.parameter_counts()
    max_length = max(len(example["word_ids"]) for example in training.examples)
    setup = {"config": config, "environment": environment(device), "parameters": counts,
             "data_sha256": sha256(args.data), "split_counts": {key: len(value) for key, value in data.items()},
             "vocab_words": len(vocab["words"]), "vocab_chars": len(vocab["chars"]),
             "embedding_coverage": coverage, "receptive_field_tokens": model.receptive_field,
             "max_training_tokens": max_length, "source_sha256": source_hashes(),
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    json_write(run_dir / "setup.json", setup)
    print(json.dumps(setup, indent=2), flush=True)
    embedding_parameters = list(model.word_embedding.parameters())
    other_parameters = [parameter for name, parameter in model.named_parameters()
                        if not name.startswith("word_embedding.") and parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": [p for p in embedding_parameters if p.requires_grad],
         "lr": tc["learning_rate"] * tc["embedding_lr_multiplier"]},
        {"params": other_parameters, "lr": tc["learning_rate"]},
    ], weight_decay=tc["weight_decay"])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    word_counts = torch.tensor(vocab["word_counts"], dtype=torch.float32, device=device)
    alpha = tc["word_dropout_alpha"]
    word_dropout = alpha / (word_counts + alpha) if alpha else torch.zeros_like(word_counts)
    word_dropout[:2] = 0
    batches_per_epoch = math.ceil(len(training) / tc["batch_size"])
    total_steps = batches_per_epoch * tc["epochs"]
    warmup_steps = batches_per_epoch * tc["warmup_epochs"]
    history, best, start_epoch, global_step, previous_elapsed = [], None, 1, 0, 0.0
    if args.resume:
        saved = torch.load(run_dir / "last.pt", map_location=device, weights_only=True)
        if saved["config"] != config or saved["data_sha256"] != setup["data_sha256"]:
            raise ValueError("Resume configuration/data differs from checkpoint")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        history, best = saved["history"], saved["best"]
        start_epoch, global_step = saved["epoch"] + 1, saved["global_step"]
        previous_elapsed = saved["elapsed_seconds"]
    started = time.perf_counter()
    for epoch in range(start_epoch, tc["epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train()
        intent_sum = slot_sum = 0.0
        seen = seen_tokens = 0
        order = epoch_batches(training.examples, tc["batch_size"], tc["seed"] + epoch,
                              tc["bucket_multiplier"])
        for indices in order:
            batch = move_batch(collate_batch([training.examples[i] for i in indices]), device)
            if alpha:
                dropped = torch.rand(batch["word_ids"].shape, device=device) < word_dropout[batch["word_ids"]]
                batch["word_ids"] = batch["word_ids"].masked_fill(dropped, vocab["unk_id"])
            scale = learning_rate_scale(global_step, total_steps, warmup_steps, tc["min_lr_ratio"])
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * scale
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["word_ids"], batch["char_ids"], batch["lengths"])
            loss, components = model.compute_loss(*outputs, batch["intent_ids"], batch["slot_ids"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {global_step}")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), tc["clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            amount = len(indices)
            token_count = int(batch["lengths"].sum())
            slot_denominator = token_count if model.config["slot_loss_normalization"] == "token" else amount
            intent_sum += components["intent_loss"].item() * amount
            slot_sum += components["slot_loss"].item() * slot_denominator
            seen += amount
            seen_tokens += token_count
            global_step += 1
        assert seen == 11514
        training_seconds = time.perf_counter() - epoch_started
        dev = evaluate(model, validation, vocab, device, tc["eval_batch_size"], include_loss=True)
        epoch_seconds = time.perf_counter() - epoch_started
        elapsed = previous_elapsed + time.perf_counter() - started
        train_intent = intent_sum / seen
        denominator = seen_tokens if model.config["slot_loss_normalization"] == "token" else seen
        train_slot = slot_sum / denominator
        row = {"epoch": epoch, "train_loss": train_intent + model.config["slot_weight"] * train_slot,
               "train_intent_loss": train_intent, "train_slot_loss": train_slot,
               "dev_loss": dev["loss"], "dev_intent_acc": dev["intent_acc"],
               "dev_slot_f1": dev["slot_f1"], "dev_combined": dev["combined"],
               "dev_intent_correct": dev["intent_correct"], "dev_tp": dev["tp"],
               "dev_fp": dev["fp"], "dev_fn": dev["fn"], "learning_rate": optimizer.param_groups[-1]["lr"],
               "training_seconds": training_seconds, "epoch_seconds": epoch_seconds,
               "elapsed_seconds": elapsed}
        history.append(row)
        improved = best is None or dev["combined"] > best["validation"]["combined"]
        if improved:
            best = {"epoch": epoch, "validation": dev}
            torch.save({"state_dict": model.state_dict(), "vocab": vocab, "config": config,
                        "epoch": epoch, "validation": dev, "parameters": counts,
                        "data_sha256": setup["data_sha256"], "source_sha256": source_hashes()}, run_dir / "best.pt")
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "config": config, "data_sha256": setup["data_sha256"], "history": history,
                    "best": best, "epoch": epoch, "global_step": global_step,
                    "elapsed_seconds": elapsed}, run_dir / "last.pt")
        print(f"Epoch {epoch:02d}/{tc['epochs']} | train {row['train_loss']:.4f} | dev {dev['loss']:.4f} | "
              f"intent {dev['intent_acc']:.2f}% | slot {dev['slot_f1']:.2f}% | combined {dev['combined']:.2f}% | "
              f"{epoch_seconds:.1f}s" + (" | BEST" if improved else ""), flush=True)
        if epoch >= tc["minimum_epochs"] and epoch - best["epoch"] >= tc["patience"]:
            print(f"Early stopping: no validation improvement for {tc['patience']} epochs", flush=True)
            break
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    verified_dev = evaluate(model, validation, vocab, device, tc["eval_batch_size"], include_loss=True,
                            save_predictions=run_dir / "dev_predictions.jsonl")
    for key in ["n", "intent_correct", "tp", "fp", "fn"]:
        assert verified_dev[key] == best["validation"][key], (key, verified_dev, best)
    summary = {"name": config["name"], "seed": tc["seed"], "best_epoch": best["epoch"],
               "epochs_trained": history[-1]["epoch"], "parameters": counts,
               "validation": verified_dev, "elapsed_seconds": history[-1]["elapsed_seconds"],
               "checkpoint": str((run_dir / "best.pt").resolve()),
               "checkpoint_sha256": sha256(run_dir / "best.pt"),
               "test_evaluated": False, "reload_verified": True}
    json_write(run_dir / "summary.json", summary)
    print("TRAINING COMPLETE\n" + json.dumps(summary, indent=2), flush=True)
    return summary


def load_run(path, device):
    saved = torch.load(Path(path) / "best.pt", map_location=device, weights_only=True)
    model = build_model(saved["config"]["model"], saved["vocab"], device=device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


def select(args):
    """Compare prespecified member runs and their full ensemble on dev only."""
    device = configure(42, args.threads)
    raw = load_splits(args.data)
    runs = [load_run(path, device) for path in args.run_dirs]
    vocab = runs[0][1]["vocab"]
    data_hash = sha256(args.data)
    for _, saved in runs:
        assert saved["vocab"] == vocab
        assert saved["data_sha256"] == data_hash
    dev_dataset = MassiveDataset(raw["dev"], vocab)
    candidates = [(f"member_{i}", [i]) for i in range(len(runs))]
    if len(runs) > 1:
        candidates.append(("ensemble", list(range(len(runs)))))
    scores = []
    for name, indices in candidates:
        models = [runs[i][0] for i in indices]
        total = sum(model.parameter_counts()["total"] for model in models)
        if total > 20_000_000:
            print(f"Excluded {name}: {total:,} parameters exceeds cap", flush=True)
            continue
        metrics = evaluate(models, dev_dataset, vocab, device, args.batch_size)
        scores.append({"name": name, "members": indices, "validation": metrics,
                       "parameters": {key: sum(model.parameter_counts()[key] for model in models)
                                      for key in ("total", "trainable")}})
        print(f"{name}: intent {metrics['intent_acc']:.2f}% | slot {metrics['slot_f1']:.2f}% | "
              f"combined {metrics['combined']:.2f}% | {total:,} parameters", flush=True)
    selected = max(scores, key=lambda score: (score["validation"]["combined"], -score["parameters"]["total"]))
    manifest = {"selected": selected, "candidates": scores,
                "run_dirs": [str(Path(path).resolve()) for path in args.run_dirs],
                "checkpoints": [{"path": str((Path(path) / 'best.pt').resolve()),
                                 "sha256": sha256(Path(path) / "best.pt")} for path in args.run_dirs],
                "data_sha256": data_hash, "test_evaluated": False,
                "selection_rule": "maximum dev combined score; ties prefer fewer parameters",
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": source_hashes()}
    json_write(Path(args.output) / "selection.json", manifest)
    print("FROZEN SELECTION\n" + json.dumps(selected, indent=2), flush=True)
    return manifest


def final(args):
    directory = Path(args.output)
    if (directory / "results.json").exists() and not args.verify:
        raise ValueError("Final results already exist; use --verify to reproduce frozen evaluation")
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    assert sha256(args.data) == selection["data_sha256"]
    # The final checkpoint set is frozen before any test prediction is made.
    for checkpoint in selection["checkpoints"]:
        assert sha256(checkpoint["path"]) == checkpoint["sha256"]
    device = configure(42, args.threads)
    runs = [load_run(selection["run_dirs"][i], device) for i in selection["selected"]["members"]]
    models = [run[0] for run in runs]
    vocab = runs[0][1]["vocab"]
    raw = load_splits(args.data)
    validation = evaluate(models, MassiveDataset(raw["dev"], vocab), vocab, device,
                          args.batch_size, save_predictions=directory / "dev_predictions.jsonl")
    for key in ["n", "intent_correct", "tp", "fp", "fn"]:
        assert validation[key] == selection["selected"]["validation"][key]
    test = evaluate(models, MassiveDataset(raw["test"], vocab), vocab, device,
                    args.batch_size, save_predictions=directory / "test_predictions.jsonl")
    summaries = [json.loads((Path(selection["run_dirs"][i]) / "summary.json").read_text(encoding="utf-8"))
                 for i in selection["selected"]["members"]]
    results = {"system": "shared residual dilated CNN with joint intent head and BIO-constrained CRF",
               "selected_candidate": selection["selected"]["name"],
               "parameters": selection["selected"]["parameters"],
               "validation": validation, "test": test,
               "best_epochs": [summary["best_epoch"] for summary in summaries],
               "epochs_trained": [summary["epochs_trained"] for summary in summaries],
               "training_seconds_selected_members": sum(summary["elapsed_seconds"] for summary in summaries),
               "environment": environment(device), "reload_verified": True,
               "selection_sha256": sha256(directory / "selection.json"),
               "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
               "disclosure": "AI-assisted implementation and execution; not an independent Stage-1 baseline."}
    if args.verify:
        previous = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        for split in ("validation", "test"):
            for key in ("n", "intent_correct", "tp", "fp", "fn"):
                assert previous[split][key] == results[split][key]
        json_write(directory / "verification.json", {"exact_counts_reproduced": True,
                   "verified_at_utc": results["evaluated_at_utc"], "results": results})
    else:
        json_write(directory / "results.json", results)
    print("FINAL RESULTS\n" + json.dumps(results, indent=2), flush=True)
    return results


def benchmark(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = configure(config["training"]["seed"], args.threads)
    raw = load_splits(args.data)
    vocab = build_vocab(raw["train"])
    dataset = MassiveDataset(raw["train"], vocab)
    model = build_model(config["model"], vocab, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    examples = sorted(dataset.examples, key=lambda ex: len(ex["word_ids"]))
    # Take representative middle-length examples; timing is not a quality result.
    batch = move_batch(collate_batch(examples[len(examples)//2:len(examples)//2 + args.batch_size]), device)
    step_times = []
    model.train()
    for step in range(args.steps + 3):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(*model(batch["word_ids"], batch["char_ids"], batch["lengths"]),
                                     batch["intent_ids"], batch["slot_ids"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step >= 3:
            step_times.append(time.perf_counter() - started)
    result = {"threads": args.threads, "batch_size": args.batch_size,
              "batch_tokens": int(batch["lengths"].sum()), "parameters": model.parameter_counts(),
              "median_training_step_seconds": float(np.median(step_times)),
              "rough_epoch_training_seconds": float(np.median(step_times)) * math.ceil(11514 / args.batch_size),
              "note": "Timing pilot only, random embeddings, no reported model quality, no test evaluation"}
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--config", default=str(ROOT / "configs" / "cnn192.json"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--glove", default=str(DEFAULT_GLOVE))
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(function=train)
    p = sub.add_parser("select")
    p.add_argument("--run-dirs", nargs="+", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--output", default=str(ROOT / "results"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.set_defaults(function=select)
    p = sub.add_parser("final")
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--output", default=str(ROOT / "results"))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--verify", action="store_true")
    p.set_defaults(function=final)
    p = sub.add_parser("benchmark")
    p.add_argument("--config", default=str(ROOT / "configs" / "cnn192.json"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=12)
    p.set_defaults(function=benchmark)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
