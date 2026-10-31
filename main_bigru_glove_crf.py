"""Train a GloVe + bidirectional GRU + CRF with dev early stopping."""

import argparse
from pathlib import Path

from main_glove_crf import run
from sc4002.models.joint_bigru_glove_crf import JointBiGRUGloveCRF


def model_config(args, vocab, intents, tags):
    if min(args.embedding_dim, args.hidden_size, args.gru_layers) < 1:
        raise ValueError("BiGRU dimensions and layer count must be positive")
    return dict(
        vocab_size=len(vocab), glove_dim=args.glove_dim, slot_tags=tags,
        num_intents=len(intents), embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size, num_layers=args.gru_layers,
        dropout=args.dropout, max_words=128, pad_id=0,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--dev-csv", type=Path, default=Path("data/dev.csv"))
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--glove-dim", type=int, default=100)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("checkpoints/bigru_glove_crf"))
    parser.add_argument("--log-dir", type=Path,
                        default=Path("logs/bigru_glove_crf"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/bigru_glove_crf"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--slot-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args(), model_class=JointBiGRUGloveCRF,
        model_type="glove_bigru_crf", config_factory=model_config)
