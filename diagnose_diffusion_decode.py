"""Compare one-step and iterative slot diffusion decoding on dev."""

import argparse
from pathlib import Path

import torch

from main import read_split
from main_glove_crf import make_loader
from sc4002.eval import EvalMetric, SpanDecoder
from sc4002.models.joint_bigru_glove_diffusion import JointBiGRUGloveDiffusion
from sc4002.tokenizers.glove_words import GloveLookup


@torch.inference_mode()
def score(model, loader, examples, intents, tags, device, method):
    model.eval()
    decoder = SpanDecoder()
    pred_intents, pred_slots = [], []
    offset = 0
    for ids, vectors, _, _, mask in loader:
        logits, paths = getattr(model, method)(ids.to(device), vectors.to(device),
                                                mask.to(device))
        pred_intents.extend(intents[i] for i in logits.argmax(-1).tolist())
        for row, path in enumerate(paths):
            pred_slots.append(decoder.decode(
                examples[offset + row].utt, [tags[i] for i in path]
            ))
        offset += len(ids)
    metric = EvalMetric([ex.intent for ex in examples],
                        [ex.annot_utt for ex in examples])
    return metric.evaluate(pred_intents, pred_slots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--glove", type=Path, default=Path("data/glove.6B.zip"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    dev = read_split(Path("data/dev.csv"), "dev", 2_033)
    lookup = GloveLookup(args.glove, (word for ex in dev for word in ex.words),
                         saved["glove_dim"])
    loader = make_loader(dev, saved["vocab"], lookup, saved["intents"],
                         saved["tags"], 64)
    model = JointBiGRUGloveDiffusion(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model_state"])
    for method in ("predict_all_mask", "predict"):
        scores = score(model, loader, dev, saved["intents"], saved["tags"],
                       device, method)
        print(f"{method}: slot F1={100*scores['slot_f1']:.2f}% "
              f"intent={100*scores['intent_accuracy']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
