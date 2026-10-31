"""Train a word-level GloVe Transformer + CRF with dev early stopping."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from main import build_label_maps, print_scores, read_split
from sc4002.eval import EvalMetric, SpanDecoder
from sc4002.models.joint_transformer_glove_crf import JointTransformerGloveCRF
from sc4002.tokenizers.glove_words import (
    GloveLookup, GloveWordDataset, build_word_vocab, collate_glove,
)


def make_loader(examples, vocab, lookup, intents, tags, batch_size, *, shuffle=False):
    return DataLoader(
        GloveWordDataset(examples, vocab, lookup, intents, tags),
        batch_size=batch_size, shuffle=shuffle, collate_fn=collate_glove,
    )


def train_epoch(model, loader, optimizer, device, slot_weight):
    model.train()
    total = 0.0
    for ids, vectors, intents, slots, mask in loader:
        ids, vectors = ids.to(device), vectors.to(device)
        intents, slots, mask = intents.to(device), slots.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(model(ids, vectors, mask), intents, slots, mask,
                          slot_weight=slot_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(ids)
    return total / len(loader.dataset)


@torch.inference_mode()
def evaluate_glove(model, loader, examples, intents, tags, device, slot_weight,
                   *, return_predictions=False):
    model.eval()
    decoder = SpanDecoder()
    pred_intents, pred_slots = [], []
    total = 0.0
    offset = 0
    for ids, vectors, gold_intents, gold_slots, mask in loader:
        ids, vectors, mask = ids.to(device), vectors.to(device), mask.to(device)
        output = model(ids, vectors, mask)
        loss = model.loss(output, gold_intents.to(device), gold_slots.to(device),
                          mask, slot_weight=slot_weight)
        total += loss.item() * len(ids)
        pred_intents.extend(intents[i] for i in output.intent_logits.argmax(-1).tolist())
        for row, path in enumerate(model.decode(output, mask)):
            pred_slots.append(decoder.decode(
                examples[offset + row].utt, [tags[index] for index in path],
            ))
        offset += len(ids)
    metric = EvalMetric([ex.intent for ex in examples],
                        [ex.annot_utt for ex in examples])
    result = total / len(examples), metric.evaluate(pred_intents, pred_slots)
    if return_predictions:
        return *result, pred_intents, pred_slots
    return result


def run(args, *, model_class=JointTransformerGloveCRF,
        model_type="glove_transformer_crf", config_factory=None):
    if min(args.epochs, args.patience, args.batch_size, args.glove_dim) < 1:
        raise ValueError("Training and model sizes must be positive")
    if args.lr <= 0 or args.slot_weight <= 0 or args.min_delta < 0:
        raise ValueError("Invalid learning rate, slot weight, or min delta")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train = read_split(args.train_csv, "train", 11_514)
    dev = read_split(args.dev_csv, "dev", 2_033)
    intents, tags = build_label_maps(train)
    vocab = build_word_vocab(train)
    lookup = GloveLookup(args.glove, (word for ex in train + dev for word in ex.words),
                         args.glove_dim)
    print(f"word vocabulary: {len(vocab):,}; GloVe matches: {len(lookup.vectors):,}")
    if config_factory is None:
        if min(args.d_model, args.nhead, args.num_layers, args.dim_feedforward) < 1:
            raise ValueError("Transformer sizes must be positive")
        config = dict(
            vocab_size=len(vocab), glove_dim=args.glove_dim, slot_tags=tags,
            num_intents=len(intents), d_model=args.d_model, nhead=args.nhead,
            num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
            max_words=128, pad_id=0,
        )
    else:
        config = config_factory(args, vocab, intents, tags)
    model = model_class(**config).to(device)
    print(f"parameters: {model.parameter_count():,} / 20,000,000")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = make_loader(train, vocab, lookup, intents, tags,
                               args.batch_size, shuffle=True)
    dev_loader = make_loader(dev, vocab, lookup, intents, tags, args.batch_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "best_glove_crf.pt"
    history = []
    best_score, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.slot_weight)
        dev_loss, scores = evaluate_glove(model, dev_loader, dev, intents, tags,
                                          device, args.slot_weight)
        score = scores["slot_f1"]
        history.append(dict(epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
                            dev_intent_accuracy=scores["intent_accuracy"],
                            dev_slot_micro_f1=score,
                            elapsed_seconds=time.perf_counter() - start))
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} "
              f"dev_loss={dev_loss:.4f} intent={100*scores['intent_accuracy']:.2f}% "
              f"slot_f1={100*score:.2f}%", flush=True)
        if score > best_score + args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(dict(
                model_type=model_type, model_state=model.state_dict(),
                model_config=config, tokenizer_type="glove_words", vocab=vocab,
                glove_dim=args.glove_dim, intents=intents, tags=tags,
                slot_weight=args.slot_weight, epoch=epoch, selection_score=score,
            ), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping after {args.patience} stale epochs")
                break
    (args.log_dir / "glove_crf_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch: {best_epoch}; dev slot F1: {100*best_score:.2f}%")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    best_model = model_class(**saved["model_config"]).to(device)
    best_model.load_state_dict(saved["model_state"])
    dev_loss, dev_scores = evaluate_glove(best_model, dev_loader, dev, intents,
                                          tags, device, args.slot_weight)
    print_scores("best checkpoint dev", dev_loss, dev_scores)
    if args.test_csv is not None:
        from eval_glove import evaluate_checkpoint
        evaluate_checkpoint(checkpoint, args.glove, args.test_csv,
                            args.batch_size, args.output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--glove", type=Path, required=True,
                        help="Path to glove.6B.100d.txt or glove.6B.zip")
    parser.add_argument("--glove-dim", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/glove_crf"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/glove_crf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/glove_crf"))
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
