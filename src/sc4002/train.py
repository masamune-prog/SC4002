"""Training, checkpoint selection and evaluation. Also a CLI for ablation runs:

    uv run python -m sc4002.train --exp default --seeds 42 43
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .data import DATA_DIR, ROOT, Encoder, Tokeniser, batches, collate, load_splits
from .metrics import fmt, score, tags_to_spans
from .model import JointTransformer, count_params
from .vectors import MAX_TOP_K, VectorTable, load_store

PARAM_CAP = 20_000_000
RESULTS = Path(os.environ.get("SC4002_RESULTS", ROOT / "results"))
CKPT_DIR = Path(os.environ.get("SC4002_CKPT", ROOT / "checkpoints"))


@dataclass
class Config:
    # tokenisation / inputs
    tokeniser: str = "word"  # word | clitic | bpe
    bpe_vocab: int = 2000
    use_glove: bool = True  # frozen GloVe 6B 300d table
    top_k: int = -1  # extra frequent GloVe words in the frozen table; -1 = fill the 20M budget
    d_task: int = 128  # trainable unit embedding
    char_dim: int = 32
    char_filters: int = 100  # 0 = no char-CNN
    # encoder
    d_model: int = 384
    heads: int = 6
    layers: int = 4
    ffn: int = 1536
    positions: str = "relative"  # relative | learned
    max_len: int = 128
    dropout: float = 0.2
    attn_dropout: float = 0.1
    emb_dropout: float = 0.25
    # heads
    intent_cond: bool = True
    d_intent_ctx: int = 64
    decoder: str = "crf"  # crf (BIO-constrained) | softmax
    # optimisation
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    epochs: int = 40
    batch_size: int = 32
    clip: float = 1.0
    label_smoothing: float = 0.1
    word_dropout: float = 0.25
    seed: int = 42
    device: str = "auto"


# ---------------------------------------------------------------- setup


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(cfg: Config) -> str:
    if cfg.device != "auto":
        return cfg.device
    if torch.cuda.is_available():
        return "cuda"
    return os.environ.get("SC4002_DEVICE", "cpu")


_SPLITS = None
_STORE = {}


def build(cfg: Config, verbose: bool = True):
    """Data, encoder, vector table and model for a config (deterministic given cfg.seed)."""
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = load_splits()
    splits = _SPLITS
    train = splits["train"]
    tok = Tokeniser(cfg.tokeniser, train, cfg.bpe_vocab, save_dir=DATA_DIR)
    vectors = None
    if cfg.use_glove:
        keys = {k for ex in train for w in ex.words for k in tok.vector_keys(w, tok.word_units(w))}
        tag = frozenset(keys)
        if tag not in _STORE:
            _STORE[tag] = load_store(keys)
        if cfg.top_k >= 0:
            vectors = VectorTable(_STORE[tag], keys, cfg.top_k)
        else:  # size the rest of the model with train rows only, then spend what is left on top-K rows
            base = VectorTable(_STORE[tag], keys, 0)
            rest = count_params(JointTransformer(cfg, 1, 1, ["O"], 1, base.matrix))[0] - base.matrix.size
            rest += _head_params(cfg, train, tok)
            spare = (PARAM_CAP - rest) // base.dim - base.matrix.shape[0]
            vectors = VectorTable(_STORE[tag], keys, MAX_TOP_K, extra_rows=spare)
    enc = Encoder(train, tok, vectors)
    data = {k: [enc.encode(ex) for ex in v] for k, v in splits.items()}
    set_seed(cfg.seed)
    model = JointTransformer(
        cfg, len(enc.units), len(enc.chars), enc.tags.itos, len(enc.intents),
        vectors.matrix if vectors else None)
    total, trainable = count_params(model)
    assert total <= PARAM_CAP, f"{total:,} parameters exceeds the 20M cap"
    if verbose:
        rows = vectors.matrix.shape if vectors else (0, 0)
        print(f"units {len(enc.units)}  chars {len(enc.chars)}  frozen table {rows[0]}x{rows[1]}  "
              f"params total {total:,}  trainable {trainable:,}")
    return enc, data, model


def _head_params(cfg: Config, train, tok) -> int:
    """Parameters that depend on vocab/label sizes, missing from the 1-unit probe model."""
    enc = Encoder(train, tok)
    full = JointTransformer(cfg, len(enc.units), len(enc.chars), enc.tags.itos, len(enc.intents))
    probe = JointTransformer(cfg, 1, 1, ["O"], 1)
    return count_params(full)[0] - count_params(probe)[0]


def to_device(b: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in b.items()}


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def predict(model, data, device, batch_size: int = 256):
    model.eval()
    intents, tags, loss_sum, n = [], [], 0.0, 0
    for chunk in batches(data, batch_size, shuffle=False):
        b = to_device(collate(chunk), device)
        loss, _, _ = model.loss(b)
        loss_sum += loss.item() * len(chunk)
        n += len(chunk)
        i, t = model.predict(b)
        intents += i
        tags += t
    return intents, tags, loss_sum / n


def evaluate(model, enc: Encoder, data, device) -> tuple[dict, list]:
    intents, tags, loss = predict(model, data, device)
    pred_intents = [enc.intents.itos[i] for i in intents]
    pred_spans = [tags_to_spans(e.ex.words, [enc.tags.itos[t] for t in ts]) for e, ts in zip(data, tags)]
    s = score([e.ex.intent for e in data], pred_intents, [e.ex.gold for e in data], pred_spans)
    s["loss"] = loss
    preds = [dict(id=e.ex.id, utt=e.ex.utt, gold_intent=e.ex.intent, pred_intent=pi,
                  gold_slots=e.ex.gold, pred_slots=ps)
             for e, pi, ps in zip(data, pred_intents, pred_spans)]
    return s, preds


# ---------------------------------------------------------------- training


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or name.startswith("crf") or "rel." in name or "vec" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def fit(cfg: Config, name: str = "run", save_checkpoint: bool = False, verbose: bool = True,
        built=None) -> dict:
    """Train on train, select the epoch with the best dev combined score.
    Returns a summary with the per-epoch history and best-epoch dev scores."""
    device = pick_device(cfg)
    enc, data, model = built or build(cfg, verbose)
    model.to(device)
    total, trainable = count_params(model)
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    steps_per_epoch = math.ceil(len(data["train"]) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    rng = np.random.default_rng(cfg.seed)
    history, best, best_state = [], None, None
    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, tl, tn = time.time(), 0.0, 0
        for chunk in batches(data["train"], cfg.batch_size, shuffle=True, rng=rng):
            b = to_device(collate(chunk, cfg.word_dropout, rng), device)
            loss, _, _ = model.loss(b, cfg.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            sched.step()
            tl += loss.item() * len(chunk)
            tn += len(chunk)
        dev, _ = evaluate(model, enc, data["dev"], device)
        row = dict(epoch=epoch, train_loss=tl / tn, dev_loss=dev["loss"], dev_intent_acc=dev["intent_acc"],
                   dev_slot_f1=dev["slot_f1"], dev_combined=dev["combined"],
                   epoch_time=time.time() - t0, elapsed=time.time() - start, lr=sched.get_last_lr()[0])
        history.append(row)
        improved = best is None or dev["combined"] > best["dev_combined"]
        if improved:
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:2d}  train loss {row['train_loss']:.4f}  dev loss {row['dev_loss']:.4f}  "
                  f"dev {fmt(dev)}  {row['epoch_time']:.1f}s{'  *' if improved else ''}", flush=True)
    ckpt_path = None
    if save_checkpoint:
        CKPT_DIR.mkdir(exist_ok=True)
        ckpt_path = CKPT_DIR / f"{name}.pt"
        torch.save(dict(state=best_state, config=asdict(cfg), epoch=best["epoch"],
                        dev=best, params=(total, trainable)), ckpt_path)
    summary = dict(name=name, config=asdict(cfg), best_epoch=best["epoch"],
                   dev_intent_acc=best["dev_intent_acc"], dev_slot_f1=best["dev_slot_f1"],
                   dev_combined=best["dev_combined"], total_params=total, trainable_params=trainable,
                   train_time=time.time() - start, device=device, threads=torch.get_num_threads(),
                   torch=torch.__version__, machine=platform.machine(), history=history,
                   checkpoint=str(ckpt_path) if ckpt_path else None)
    model.load_state_dict(best_state)
    summary["model"] = model
    summary["encoder"] = enc
    summary["data"] = data
    return summary


def load_checkpoint(path) -> tuple[torch.nn.Module, Encoder, dict, dict]:
    """Rebuild the model from the config stored in the checkpoint and load the saved weights."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    enc, data, model = build(cfg, verbose=False)
    model.load_state_dict(ck["state"])
    return model, enc, data, ck


# ---------------------------------------------------------------- CLI


def main():
    from .experiments import EXPERIMENTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--out", default="results/ablations.jsonl")
    args = ap.parse_args()
    out = Path(args.out) if os.path.isabs(args.out) else RESULTS / Path(args.out).name
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():  # resumable: skip (exp, seed) pairs already recorded
        done = {(r["exp"], r["config"]["seed"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    for exp in args.exp:
        for seed in args.seeds:
            if (exp, seed) in done:
                print(f"=== {exp} seed {seed}: already done, skipping")
                continue
            cfg = Config(**{**EXPERIMENTS[exp], "seed": seed})
            print(f"=== {exp} seed {seed} on {pick_device(cfg)}", flush=True)
            try:
                s = fit(cfg, name=f"{exp}_s{seed}", verbose=True)
            except AssertionError as e:  # e.g. over the parameter cap: skip, keep the grid going
                print(f"=== {exp} seed {seed}: SKIPPED ({e})", flush=True)
                continue
            for k in ("model", "encoder", "data"):
                s.pop(k)
            s["exp"] = exp
            with open(out, "a") as f:
                f.write(json.dumps(s) + "\n")
            print(f"=== {exp} seed {seed}: best epoch {s['best_epoch']}  dev combined {s['dev_combined']:.2f}  "
                  f"({s['dev_intent_acc']:.2f} / {s['dev_slot_f1']:.2f})  {s['train_time']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
