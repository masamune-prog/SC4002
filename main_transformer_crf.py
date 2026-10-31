"""Train a joint Transformer and BIO CRF on MASSIVE en-US train.csv.

Only BPE IDs and word positions derived from ``utt`` enter the model. Dev slot
micro F1 selects the checkpoint and controls early stopping. Test is evaluated
only after the best checkpoint is restored.
"""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from main import EncodedDataset, build_label_maps, print_scores, read_split
from sc4002.eval import EvalMetric, SpanDecoder
from sc4002.models import JointTransformerCRF
from sc4002.tokenizers import BpeTokenizer


def collate_crf(rows, *, pad_id: int):
    """Pack BPE IDs and original-word BIO targets into padded tensors."""
    batch_size = len(rows)
    max_tokens = max(len(encoding.ids) for encoding, _, _ in rows)
    starts = []
    for encoding, _, _ in rows:
        word_ids = encoding.word_ids
        starts.append([
            token_index for token_index, word_id in enumerate(word_ids)
            if word_id is not None and (token_index == 0 or word_id != word_ids[token_index - 1])
        ])
    max_words = max(len(indices) for indices in starts)
    input_ids = torch.full((batch_size, max_tokens), pad_id, dtype=torch.long)
    intent_labels = torch.empty(batch_size, dtype=torch.long)
    word_positions = torch.zeros((batch_size, max_words), dtype=torch.long)
    word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    slot_labels = torch.zeros((batch_size, max_words), dtype=torch.long)
    for row, ((encoding, intent, aligned_tags), positions) in enumerate(zip(rows, starts)):
        token_count = len(encoding.ids)
        word_count = len(positions)
        if word_count == 0:
            raise ValueError("Each utterance must have at least one word")
        input_ids[row, :token_count] = torch.tensor(encoding.ids)
        intent_labels[row] = intent
        word_positions[row, :word_count] = torch.tensor(positions)
        word_mask[row, :word_count] = True
        slot_labels[row, :word_count] = torch.tensor([aligned_tags[i] for i in positions])
    return input_ids, intent_labels, slot_labels, word_positions, word_mask


def train_epoch(model, loader, optimizer, device, slot_weight: float) -> float:
    model.train()
    loss_sum = 0.0
    for input_ids, intents, slots, positions, word_mask in loader:
        input_ids = input_ids.to(device)
        intents = intents.to(device)
        slots = slots.to(device)
        positions = positions.to(device)
        word_mask = word_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, positions, word_mask)
        loss = model.loss(output, intents, slots, word_mask, slot_weight=slot_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(input_ids)
    return loss_sum / len(loader.dataset)


@torch.inference_mode()
def evaluate_crf(
    model, loader, examples, intents, tags, device, slot_weight: float,
    *, return_predictions: bool = False,
):
    model.eval()
    decoder = SpanDecoder()
    predicted_intents = []
    predicted_slots = []
    loss_sum = 0.0
    offset = 0
    for input_ids, gold_intents, gold_slots, positions, word_mask in loader:
        word_mask = word_mask.to(device)
        output = model(input_ids.to(device), positions.to(device), word_mask)
        loss = model.loss(
            output, gold_intents.to(device), gold_slots.to(device), word_mask,
            slot_weight=slot_weight,
        )
        loss_sum += loss.item() * len(input_ids)
        intent_ids = output.intent_logits.argmax(-1).tolist()
        slot_paths = model.decode(output, word_mask)
        predicted_intents.extend(intents[index] for index in intent_ids)
        for row, path in enumerate(slot_paths):
            word_tags = [tags[index] for index in path]
            predicted_slots.append(decoder.decode(examples[offset + row].utt, word_tags))
        offset += len(input_ids)
    metric = EvalMetric(
        [example.intent for example in examples],
        [example.annot_utt for example in examples],
    )
    result = loss_sum / len(examples), metric.evaluate(predicted_intents, predicted_slots)
    if return_predictions:
        return *result, predicted_intents, predicted_slots
    return result


def run(args) -> None:
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if args.lr <= 0 or args.min_delta < 0 or args.slot_weight <= 0:
        raise ValueError("lr and slot_weight must be positive; min_delta must be non-negative")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train = read_split(args.train_csv, "train", 11_514)
    dev = read_split(args.dev_csv, "dev", 2_033)
    tokenizer = BpeTokenizer.load(args.tokenizer)
    intents, tags = build_label_maps(train)
    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "num_intents": len(intents),
        "slot_tags": tags,
        "d_model": 384,
        "nhead": 6,
        "num_layers": 9,
        "dim_feedforward": 1536,
        "norm_first": True,
        "max_words": 128,
        "pad_id": tokenizer.pad_id,
    }
    model = JointTransformerCRF(**model_config).to(device)
    print(f"BPE vocabulary: {tokenizer.vocab_size:,}; BIO tags: {len(tags)}")
    print(f"trainable parameters: {model.parameter_count():,} / 20,000,000")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    collate = partial(collate_crf, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        EncodedDataset(train, tokenizer, intents, tags),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
    )
    dev_loader = DataLoader(
        EncodedDataset(dev, tokenizer, intents, tags),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / "best_transformer_crf.pt"
    history = []
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.slot_weight)
        dev_loss, scores = evaluate_crf(
            model, dev_loader, dev, intents, tags, device, args.slot_weight
        )
        score = scores["slot_f1"]
        elapsed = time.perf_counter() - started
        history.append({
            "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
            "dev_intent_accuracy": scores["intent_accuracy"],
            "dev_slot_micro_f1": score, "elapsed_seconds": elapsed,
        })
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}"
            f"  intent={100*scores['intent_accuracy']:.2f}%"
            f"  slot_f1={100*score:.2f}%  time={elapsed:.1f}s",
            flush=True,
        )
        if score > best_score + args.min_delta:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save({
                "model_type": "transformer_crf",
                "model_state": model.state_dict(),
                "model_config": model_config,
                "tokenizer_type": "bpe",
                "tokenizer_json": tokenizer.to_json(),
                "intents": intents, "tags": tags,
                "slot_weight": args.slot_weight,
                "epoch": epoch, "selection_score": score,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping: no improvement for {args.patience} epochs")
                break

    (args.log_dir / "transformer_crf_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best epoch: {best_epoch}; dev slot micro F1: {100*best_score:.2f}%")
    print(f"checkpoint: {checkpoint_path}")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    best_model = JointTransformerCRF(**saved["model_config"]).to(device)
    best_model.load_state_dict(saved["model_state"])
    dev_loss, dev_scores = evaluate_crf(
        best_model, dev_loader, dev, intents, tags, device, args.slot_weight
    )
    print_scores("best checkpoint dev", dev_loss, dev_scores)

    if args.test_csv is not None:
        test = read_split(args.test_csv, "test", 2_974)
        test_loader = DataLoader(
            EncodedDataset(test, tokenizer, intents, tags),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate,
        )
        test_loss, test_scores = evaluate_crf(
            best_model, test_loader, test, intents, tags, device, args.slot_weight
        )
        print_scores("final test", test_loss, test_scores)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/bpe_tokenizer.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/transformer_crf_deep"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/transformer_crf_deep"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
