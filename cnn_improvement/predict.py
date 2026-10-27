"""Portable checkpoint export and utterance-only inference for the frozen model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .metrics import tags_to_pairs
from .model import JointDilatedCNN
from .train import ROOT, average_crf_decode, configure, json_write, sha256


def export_frozen_system(directory=None):
    directory = Path(directory or ROOT / "results")
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    members = []
    vocab = None
    for index in selection["selected"]["members"]:
        entry = selection["checkpoints"][index]
        checkpoint = Path(entry["path"])
        if sha256(checkpoint) != entry["sha256"]:
            raise ValueError("Frozen checkpoint changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if vocab is None:
            vocab = saved["vocab"]
        assert vocab == saved["vocab"]
        members.append({"state_dict": saved["state_dict"], "model_config": saved["config"]["model"],
                        "seed": saved["config"]["training"]["seed"], "best_epoch": saved["epoch"],
                        "source_checkpoint_sha256": entry["sha256"]})
    artifact = {"format_version": 1, "architecture": "JointDilatedCNN", "vocab": vocab,
                "members": members, "parameters": results["parameters"],
                "validation": results["validation"], "test": results["test"],
                "disclosure": results["disclosure"]}
    target = directory / "model.pt"
    torch.save(artifact, target)
    exported = {"file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size,
                "parameters": results["parameters"], "members": len(members),
                "vocab_included": True, "external_embeddings_needed_for_inference": False}
    json_write(directory / "model_export.json", exported)
    print(json.dumps(exported, indent=2))
    return target


class Predictor:
    """Predict intent and slot pairs from text, without accepting any gold labels."""

    def __init__(self, checkpoint=None, threads=4, device=None):
        self.device = configure(42, threads) if device is None else torch.device(device)
        self.path = Path(checkpoint or ROOT / "results" / "model.pt")
        payload = torch.load(self.path, map_location=self.device, weights_only=True)
        if payload.get("format_version") != 1 or payload.get("architecture") != "JointDilatedCNN":
            raise ValueError("Unsupported model artifact")
        self.vocab = payload["vocab"]
        self.models = []
        for member in payload["members"]:
            model = JointDilatedCNN(
                member["model_config"], len(self.vocab["words"]), len(self.vocab["chars"]),
                len(self.vocab["intents"]), len(self.vocab["tags"]), self.vocab["tags"],
            ).to(self.device)
            model.load_state_dict(member["state_dict"])
            model.eval()
            self.models.append(model)
        actual = {key: sum(model.parameter_counts()[key] for model in self.models)
                  for key in ("total", "trainable")}
        assert actual == payload["parameters"]
        assert actual["total"] <= 20_000_000
        self.parameters = actual

    def _inputs(self, utterances):
        words = [utterance.split() for utterance in utterances]
        if any(not tokens for tokens in words):
            raise ValueError("Each utterance must contain at least one token")
        lengths = torch.tensor([len(tokens) for tokens in words], device=self.device)
        longest = int(lengths.max())
        longest_word = max(len(word) for tokens in words for word in tokens)
        word_ids = torch.zeros(len(words), longest, dtype=torch.long, device=self.device)
        char_ids = torch.zeros(len(words), longest, longest_word, dtype=torch.long, device=self.device)
        for i, tokens in enumerate(words):
            word_ids[i, :len(tokens)] = torch.tensor(
                [self.vocab["word_to_id"].get(word, self.vocab["unk_id"]) for word in tokens],
                device=self.device,
            )
            for j, word in enumerate(tokens):
                char_ids[i, j, :len(word)] = torch.tensor(
                    [self.vocab["char_to_id"].get(character, self.vocab["unk_id"]) for character in word],
                    device=self.device,
                )
        return word_ids, char_ids, lengths

    @torch.inference_mode()
    def predict(self, utterances, batch_size=128):
        if isinstance(utterances, str):
            utterances = [utterances]
        utterances = list(utterances)
        if any(not isinstance(text, str) for text in utterances):
            raise TypeError("Inference accepts utterance strings only")
        predictions = []
        for start in range(0, len(utterances), batch_size):
            chunk = utterances[start:start + batch_size]
            inputs = self._inputs(chunk)
            outputs = [model(*inputs) for model in self.models]
            probabilities = torch.stack([output[0].softmax(-1) for output in outputs]).mean(0)
            emissions = torch.stack([output[1] for output in outputs]).mean(0)
            mask = outputs[0][2]
            ids = (self.models[0].decode(emissions, mask) if len(self.models) == 1 else
                   average_crf_decode(self.models, emissions, mask))
            intent_ids = probabilities.argmax(-1).cpu().tolist()
            for utterance, intent_id, length, row in zip(chunk, intent_ids, inputs[2].cpu().tolist(), ids.cpu().tolist()):
                tags = [self.vocab["tags"][tag] for tag in row[:length]]
                predictions.append({"utt": utterance, "intent": self.vocab["intents"][intent_id],
                                    "slots": sorted(tags_to_pairs(utterance, tags))})
        return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT / "results" / "model.pt"))
    parser.add_argument("--text", action="append", help="Utterance to predict; may be specified repeatedly")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--export", action="store_true", help="Export the already frozen/evaluated system")
    args = parser.parse_args()
    if args.export:
        export_frozen_system(Path(args.checkpoint).parent)
    if args.text:
        predictor = Predictor(args.checkpoint, args.threads)
        print(json.dumps(predictor.predict(args.text), indent=2, ensure_ascii=False))
    elif not args.export:
        parser.error("Provide --text or --export")


if __name__ == "__main__":
    main()
