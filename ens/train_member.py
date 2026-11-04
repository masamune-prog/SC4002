"""Train one ensemble member on the train split; keep the checkpoint with the best dev combined score.
Writes checkpoints/ens/<tag>.pt, logs/ens/<tag>.json (epoch history) and outputs/ens/<tag>_dev.pt (dev outputs
of the selected checkpoint, used to build the ensemble). The test split is never touched here."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import core as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(C.ARCHS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-slot-lex", action="store_true")
    ap.add_argument("--no-intent-lex", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=json_value")
    args = ap.parse_args()
    over = {}
    if args.no_slot_lex:
        over["slot_lex"] = False
    if args.no_intent_lex:
        over["intent_lex"] = False
    for kv in args.set:
        k, v = kv.split("=", 1)
        over[k] = json.loads(v)
    cfg = C.make_cfg(args.arch, args.seed, **over)
    tag = args.tag or f"{args.arch}_s{args.seed}"
    for d in ("checkpoints/ens", "logs/ens", "outputs/ens"):
        os.makedirs(os.path.join(C.ROOT, d), exist_ok=True)

    t0 = time.time()
    data = C.load_data()
    labels = C.Labels(data["train"])
    feats = C.build_features(data, labels)
    train_enc = C.encode(data["train"], feats, labels, is_train=True)
    dev_batches = C.make_batches(C.encode(data["dev"], feats, labels, is_train=False), 256, shuffle=False)
    print(f"[{tag}] features ready in {time.time() - t0:.0f}s | device {C.DEVICE} | config {json.dumps(cfg)}", flush=True)

    C.set_seed(args.seed)
    frozen = C.make_frozen(feats["table"]).to(C.DEVICE)
    model = C.JointNet(cfg, feats, labels, frozen).to(C.DEVICE)
    total, trainable = C.count_params(model)
    print(f"[{tag}] params total {total:,} | trainable {trainable:,}", flush=True)

    ckpt = os.path.join(C.ROOT, "checkpoints/ens", f"{tag}.pt")
    hist, best = C.fit(model, cfg, train_enc, dev_batches, data["dev"], labels, ckpt,
                       log=lambda s: print(f"[{tag}] {s}", flush=True))
    model.load_state_dict(torch.load(ckpt, map_location=C.DEVICE, weights_only=False)["state"])
    probs, spans, loss = C.member_outputs(model, dev_batches)
    res = C.evaluate_outputs(probs, spans, data["dev"], labels)
    assert res["intent_correct"] == best["intent_correct"] and res["tp"] == best["tp"], "reload mismatch"
    torch.save({"probs": probs, "spans": spans, "dev": res}, os.path.join(C.ROOT, "outputs/ens", f"{tag}_dev.pt"))
    json.dump({"tag": tag, "config": cfg, "history": hist, "best": best, "params": [total, trainable]},
              open(os.path.join(C.ROOT, "logs/ens", f"{tag}.json"), "w"), indent=1)
    print(f"[{tag}] SELECTED epoch {best['epoch']} ({best['kind']}) | dev intent {res['intent_acc']:.2f} | "
          f"slot F1 {res['slot_f1']:.2f} | combined {res['combined']:.2f} | {best['train_time_s'] / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
