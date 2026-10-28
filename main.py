"""Train the AI-assisted joint model with train-fitted BPE on MASSIVE en-US.

Only ``utt`` becomes model input. ``intent`` and ``annot_utt`` provide targets
and evaluation labels. The BPE tokenizer and label maps come from train.csv only.
Dev selects the checkpoint and controls early stopping; test is evaluated once
after the best checkpoint is restored.
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sc4002.eval import EvalMetric, SpanDecoder  # noqa: E402
from sc4002.eval.eval_helper import SLOT_PATTERN  # noqa: E402
from sc4002.models import JointTransformer  # noqa: E402
from sc4002.tokenizers import BpeTokenizer  # noqa: E402


@dataclass(frozen=True)
class Example:
    record_id: str
    utt: str
    intent: str
    annot_utt: str
    words: tuple[str, ...]
    bio_tags: tuple[str, ...]


def word_bio_tags(utterance: str, annotation: str) -> tuple[str, ...]:
    """Read MASSIVE spans and check that they reproduce the input words."""
    words: list[str] = []
    tags: list[str] = []
    position = 0
    for match in SLOT_PATTERN.finditer(annotation):
        plain_words = annotation[position:match.start()].split()
        words.extend(plain_words)
        tags.extend(["O"] * len(plain_words))
        slot_type, value = match.group(1).strip(), match.group(2)
        slot_words = value.split()
        words.extend(slot_words)
        tags.extend([f"B-{slot_type}"] + [f"I-{slot_type}"] * (len(slot_words) - 1))
        position = match.end()
    plain_words = annotation[position:].split()
    words.extend(plain_words)
    tags.extend(["O"] * len(plain_words))
    if words != utterance.split():
        raise ValueError(f"annot_utt does not align with utt: {utterance!r}")
    return tuple(tags)


def read_split(path: Path, partition: str, expected_size: int) -> list[Example]:
    examples = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"id", "utt", "intent", "annot_utt", "partition", "locale"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing required columns")
        for row_number, row in enumerate(reader, start=2):
            if row["partition"] != partition or row["locale"] != "en-US":
                raise ValueError(f"{path}:{row_number} is not en-US {partition}")
            utterance = row["utt"]
            words = tuple(utterance.split())
            if not words:
                raise ValueError(f"{path}:{row_number} has an empty utt")
            examples.append(Example(
                row["id"], utterance, row["intent"], row["annot_utt"], words,
                word_bio_tags(utterance, row["annot_utt"]),
            ))
    if len(examples) != expected_size:
        raise ValueError(f"Expected {expected_size} {partition} examples, got {len(examples)}")
    print(f"{partition}: {len(examples):,} examples")
    return examples


def build_label_maps(train: list[Example]) -> tuple[list[str], list[str]]:
    """Build intent and BIO tag IDs from the training split only."""
    intents = sorted({example.intent for example in train})
    slot_types = sorted({tag[2:] for example in train for tag in example.bio_tags if tag != "O"})
    tags = ["O"] + [tag for slot in slot_types for tag in (f"B-{slot}", f"I-{slot}")]
    if len(intents) != 60 or len(slot_types) != 55:
        raise ValueError("Expected 60 intents and 55 slot types in the training split")
    return intents, tags


class EncodedDataset(Dataset):
    def __init__(
        self, examples: list[Example], tokenizer: BpeTokenizer,
        intents: list[str], tags: list[str],
    ) -> None:
        self.examples = examples
        intent_ids = {name: i for i, name in enumerate(intents)}
        tag_ids = {name: i for i, name in enumerate(tags)}
        self.rows = []
        for example in examples:
            encoding = tokenizer.encode(example.utt)
            if not encoding.ids:
                raise ValueError(f"Tokenizer produced no IDs for {example.utt!r}")
            word_labels = [tag_ids[tag] for tag in example.bio_tags]
            aligned_labels = tokenizer.align_word_labels(encoding, word_labels)
            self.rows.append((encoding, intent_ids[example.intent], aligned_labels))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate(rows, *, pad_id=0):
    longest = max(len(encoding.ids) for encoding, _, _ in rows)
    input_ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
    slot_labels = torch.full((len(rows), longest), -100, dtype=torch.long)
    intent_labels = torch.empty(len(rows), dtype=torch.long)
    encodings = []
    for index, (encoding, intent, tags) in enumerate(rows):
        length = len(encoding.ids)
        input_ids[index, :length] = torch.tensor(encoding.ids)
        slot_labels[index, :length] = torch.tensor(tags)
        intent_labels[index] = intent
        encodings.append(encoding)
    return input_ids, intent_labels, slot_labels, encodings


def joint_loss(output, intent_labels, slot_labels):
    """Use labels as targets outside the model's prediction path."""
    intent_loss = F.cross_entropy(output.intent_logits, intent_labels)
    slot_loss_sum = F.cross_entropy(
        output.slot_logits.reshape(-1, output.slot_logits.size(-1)),
        slot_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    slot_loss = slot_loss_sum / slot_labels.ne(-100).sum().clamp_min(1)
    return intent_loss + slot_loss


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    loss_sum = 0.0
    for input_ids, intents, slots, _ in loader:
        input_ids = input_ids.to(device)
        intents = intents.to(device)
        slots = slots.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids)
        loss = joint_loss(output, intents, slots)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += loss.item() * len(input_ids)
    return loss_sum / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model, loader, examples, tokenizer, intents, tags, device,
    *, return_predictions=False,
):
    model.eval()
    decoder = SpanDecoder()
    predicted_intents = []
    predicted_slots = []
    loss_sum = 0.0
    offset = 0
    for input_ids, gold_intents, gold_slots, encodings in loader:
        output = model(input_ids.to(device))
        loss = joint_loss(output, gold_intents.to(device), gold_slots.to(device))
        loss_sum += loss.item() * len(input_ids)
        intent_ids = output.intent_logits.argmax(-1).tolist()
        slot_ids = output.slot_logits.argmax(-1).tolist()
        predicted_intents.extend(intents[i] for i in intent_ids)
        for index, encoding in enumerate(encodings):
            word_label_ids = tokenizer.collapse_subword_labels(
                encoding, slot_ids[index][:len(encoding.ids)]
            )
            word_tags = [tags[i] for i in word_label_ids]
            predicted_slots.append(decoder.decode(examples[offset + index].utt, word_tags))
        offset += len(input_ids)
    metric = EvalMetric(
        [example.intent for example in examples],
        [example.annot_utt for example in examples],
    )
    result = loss_sum / len(examples), metric.evaluate(predicted_intents, predicted_slots)
    if return_predictions:
        return *result, predicted_intents, predicted_slots
    return result


def print_scores(split: str, loss: float, scores: dict[str, float]) -> None:
    print(
        f"{split}: loss={loss:.4f}  intent_accuracy={100*scores['intent_accuracy']:.2f}%"
        f"  slot_micro_f1={100*scores['slot_f1']:.2f}%"
    )


def run(args) -> None:
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive")
    if args.lr <= 0 or args.min_delta < 0:
        raise ValueError("lr must be positive and min_delta must be non-negative")
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
    print(f"BPE vocabulary: {tokenizer.vocab_size:,} subtokens; intents: {len(intents)}; BIO tags: {len(tags)}")

    model_config = {
        "vocab_size": tokenizer.vocab_size, "num_intents": len(intents),
        "num_slot_tags": len(tags), "max_words": 128,  # BPE sequence length
        "pad_id": tokenizer.pad_id,
    }
    model = JointTransformer(**model_config).to(device)
    print(f"trainable parameters: {model.parameter_count():,} / 20,000,000")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loader = DataLoader(
        EncodedDataset(train, tokenizer, intents, tags),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=partial(collate, pad_id=tokenizer.pad_id),
    )
    dev_loader = DataLoader(
        EncodedDataset(dev, tokenizer, intents, tags),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=partial(collate, pad_id=tokenizer.pad_id),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_bpe_transformer.pt"
    history = []
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        dev_loss, scores = evaluate(model, dev_loader, dev, tokenizer, intents, tags, device)
        score = (scores["intent_accuracy"] + scores["slot_f1"]) / 2
        elapsed = time.perf_counter() - start
        history.append({
            "epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
            "dev_intent_accuracy": scores["intent_accuracy"],
            "dev_slot_micro_f1": scores["slot_f1"],
            "selection_score": score, "elapsed_seconds": elapsed,
        })
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f}  dev_loss={dev_loss:.4f}"
            f"  intent={100*scores['intent_accuracy']:.2f}%"
            f"  slot_f1={100*scores['slot_f1']:.2f}%  time={elapsed:.1f}s",
            flush=True,
        )
        if score > best_score + args.min_delta:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "model_config": model_config,
                "tokenizer_type": "bpe", "tokenizer_json": tokenizer.to_json(),
                "intents": intents, "tags": tags,
                "epoch": epoch, "selection_score": score,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping: no improvement for {args.patience} epochs")
                break

    (args.log_dir / "bpe_training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(f"best epoch: {best_epoch}; dev selection score: {100*best_score:.2f}%")
    print(f"checkpoint: {checkpoint_path}")

    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    best_model = JointTransformer(**saved["model_config"]).to(device)
    best_model.load_state_dict(saved["model_state"])
    dev_loss, dev_scores = evaluate(best_model, dev_loader, dev, tokenizer, intents, tags, device)
    print_scores("best checkpoint dev", dev_loss, dev_scores)

    if args.test_csv is not None:
        test = read_split(args.test_csv, "test", 2_974)
        test_loader = DataLoader(
            EncodedDataset(test, tokenizer, intents, tags),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=partial(collate, pad_id=tokenizer.pad_id),
        )
        test_loss, test_scores = evaluate(
            best_model, test_loader, test, tokenizer, intents, tags, device
        )
        print_scores("final test", test_loss, test_scores)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/bpe_tokenizer.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/bpe_transformer"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs/bpe_transformer"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
