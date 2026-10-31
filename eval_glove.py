"""Evaluate a frozen GloVe Transformer or BiGRU + CRF checkpoint on test.csv."""

import argparse
from pathlib import Path

import torch

from eval import save_predictions, save_results
from main import read_split
from main_glove_crf import evaluate_glove, make_loader
from sc4002.models.joint_bigru_glove_crf import JointBiGRUGloveCRF
from sc4002.models.joint_transformer_glove_crf import JointTransformerGloveCRF
from sc4002.tokenizers.glove_words import GloveLookup


def evaluate_checkpoint(checkpoint_path: Path, glove_path: Path, test_csv: Path,
                        batch_size: int, output_dir: Path):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_classes = {
        "glove_transformer_crf": JointTransformerGloveCRF,
        "glove_bigru_crf": JointBiGRUGloveCRF,
    }
    model_class = model_classes.get(saved.get("model_type"))
    if model_class is None:
        raise ValueError("Expected a GloVe Transformer or BiGRU + CRF checkpoint")
    test = read_split(test_csv, "test", 2_974)
    lookup = GloveLookup(glove_path, (word for ex in test for word in ex.words),
                         saved["glove_dim"])
    loader = make_loader(test, saved["vocab"], lookup, saved["intents"],
                         saved["tags"], batch_size)
    model = model_class(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    loss, scores, intents, slots = evaluate_glove(
        model, loader, test, saved["intents"], saved["tags"], device,
        saved["slot_weight"], return_predictions=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(output_dir / "test_predictions.csv", test, intents, slots)
    save_results(output_dir / "test_results.csv", checkpoint_path, saved, model,
                 test, loss, scores)
    print(f"checkpoint: {checkpoint_path}")
    print(f"test intent accuracy: {100*scores['intent_accuracy']:.2f}%")
    print(f"test slot micro F1: {100*scores['slot_f1']:.2f}%")
    print(f"outputs: {output_dir}")
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/glove_crf/best_glove_crf.pt"))
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/glove_crf"))
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.glove, args.test_csv,
                        args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
