"""Train a GloVe + BiGRU model with masked diffusion over BIO slot tags."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from main import build_label_maps, print_scores, read_split
from main_glove_crf import make_loader
from sc4002.eval import EvalMetric, SpanDecoder
from sc4002.models.joint_bigru_glove_diffusion import JointBiGRUGloveDiffusion
from sc4002.tokenizers.glove_words import GloveLookup, build_word_vocab


def train_epoch(model, loader, optimizer, device, slot_weight):
    model.train()
    total = 0.0
    for ids, vectors, intents, slots, mask in loader:
        ids, vectors, mask = ids.to(device), vectors.to(device), mask.to(device)
        intents, slots = intents.to(device), slots.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(ids, vectors, mask, intents, slots,
                                   slot_weight=slot_weight)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(ids)
    return total / len(loader.dataset)


@torch.inference_mode()
def evaluate_diffusion(model, loader, examples, intents, tags, device,
                       slot_weight, *, return_predictions=False):
    model.eval()
    decoder = SpanDecoder()
    pred_intents, pred_slots = [], []
    total, offset = 0.0, 0
    for ids, vectors, gold_intents, gold_slots, mask in loader:
        ids, vectors, mask = ids.to(device), vectors.to(device), mask.to(device)
        gold_intents, gold_slots = gold_intents.to(device), gold_slots.to(device)
        loss = model.validation_loss(ids, vectors, mask, gold_intents, gold_slots,
                                     slot_weight=slot_weight)
        total += loss.item() * len(ids)
        intent_logits, paths = model.predict(ids, vectors, mask)
        pred_intents.extend(intents[i] for i in intent_logits.argmax(-1).tolist())
        for row, path in enumerate(paths):
            pred_slots.append(decoder.decode(
                examples[offset + row].utt, [tags[index] for index in path]
            ))
        offset += len(ids)
    metric = EvalMetric([ex.intent for ex in examples],
                        [ex.annot_utt for ex in examples])
    result = total / len(examples), metric.evaluate(pred_intents, pred_slots)
    if return_predictions:
        return *result, pred_intents, pred_slots
    return result


def run(args, *, model_class=JointBiGRUGloveDiffusion,
        model_type="glove_bigru_diffusion"):
    if min(args.epochs, args.patience, args.batch_size, args.glove_dim,
           args.diffusion_steps) < 1:
        raise ValueError("Training settings must be positive")
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
    config = dict(
        vocab_size=len(vocab), glove_dim=args.glove_dim, slot_tags=tags,
        num_intents=len(intents), embedding_dim=256, hidden_size=256,
        gru_layers=2, denoiser_layers=2, denoiser_heads=4,
        denoiser_ff=512, diffusion_steps=args.diffusion_steps,
        dropout=0.1, max_words=128, pad_id=0,
    )
    model = model_class(**config).to(device)
    print(f"parameters: {model.parameter_count():,} / 20,000,000")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    train_loader = make_loader(train, vocab, lookup, intents, tags,
                               args.batch_size, shuffle=True)
    dev_loader = make_loader(dev, vocab, lookup, intents, tags, args.batch_size)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "best_diffusion_glove.pt"
    history = []
    best_score, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device, args.slot_weight)
        dev_loss, scores = evaluate_diffusion(
            model, dev_loader, dev, intents, tags, device, args.slot_weight,
        )
        score = scores["slot_f1"]
        history.append(dict(
            epoch=epoch, train_loss=train_loss, dev_loss=dev_loss,
            dev_intent_accuracy=scores["intent_accuracy"],
            dev_slot_micro_f1=score,
            elapsed_seconds=time.perf_counter() - started,
        ))
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
    (args.log_dir / "diffusion_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"best epoch: {best_epoch}; dev slot F1: {100*best_score:.2f}%")
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    best_model = model_class(**saved["model_config"]).to(device)
    best_model.load_state_dict(saved["model_state"])
    dev_loss, dev_scores = evaluate_diffusion(
        best_model, dev_loader, dev, intents, tags, device, args.slot_weight,
    )
    print_scores("best checkpoint dev", dev_loss, dev_scores)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--glove-dim", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/diffusion_glove"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/diffusion_glove"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--diffusion-steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
