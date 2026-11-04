"""Build the mixed ensemble from trained members.

  python ens/ensemble.py --members glu_crf_s42 glu_crf_s43 ...            # dev only: vote-threshold sweep
  python ens/ensemble.py --members ... --threshold 0.5 --final            # freeze, reload, then evaluate test once

Selection uses the dev split only. --final reloads every member checkpoint from disk, checks that the reloaded
ensemble reproduces the selected dev score exactly, and only then scores the test split."""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import core as C


def load_member(tag, feats, labels, frozen):
    ck = torch.load(os.path.join(C.ROOT, "checkpoints/ens", f"{tag}.pt"), map_location=C.DEVICE, weights_only=False)
    m = C.JointNet(ck["config"], feats, labels, frozen).to(C.DEVICE)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def ensemble_params(members, feats):
    """Frozen table counted once; every member's own parameters; lexicon table entries as frozen values."""
    shared = {id(p) for p in members[0].frozen.parameters()}
    frozen_n = sum(p.numel() for p in members[0].frozen.parameters())
    own = sum(p.numel() for m in members for p in m.parameters() if id(p) not in shared)
    lex = feats["slot_lex"].n_entries() + feats["intent_lex"].n_entries()
    total = frozen_n + own + lex
    return total, own, {"frozen_word_vectors": frozen_n, "members_trainable": own, "lexicon_entries": lex}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--name", default="final")
    args = ap.parse_args()
    data = C.load_data()
    labels = C.Labels(data["train"])

    dev_outs = [torch.load(os.path.join(C.ROOT, "outputs/ens", f"{t}_dev.pt"), weights_only=False) for t in args.members]
    for t, o in zip(args.members, dev_outs):
        r = o["dev"]
        print(f"member {t:<24} dev intent {r['intent_acc']:.2f} | slot F1 {r['slot_f1']:.2f} | comb {r['combined']:.2f}")
    probs = [o["probs"] for o in dev_outs]
    spans = [o["spans"] for o in dev_outs]
    sweep = {}
    for thr in (0.3, 0.4, 0.5, 0.6):
        p, s = C.MixedEnsemble(threshold=thr).combine(probs, spans)
        r = C.evaluate_outputs(p, s, data["dev"], labels)
        sweep[thr] = r
        print(f"ensemble({len(args.members)}) vote>={thr:.1f} | dev intent {r['intent_acc']:.2f} | "
              f"slot F1 {r['slot_f1']:.2f} | comb {r['combined']:.2f}")
    if not args.final:
        return

    thr = args.threshold if args.threshold is not None else max(sweep, key=lambda t: sweep[t]["combined"])
    selected = sweep[thr]
    print(f"\nSelected vote threshold {thr} (dev). Freezing and reloading members from checkpoints ...")
    feats = C.build_features(data, labels)
    frozen = C.make_frozen(torch.zeros_like(feats["table"])).to(C.DEVICE)      # values come from the checkpoints
    members, cks = zip(*[load_member(t, feats, labels, frozen) for t in args.members])
    total, trainable, parts = ensemble_params(list(members), feats)
    print(f"Pipeline parameters: total {total:,} | trainable {trainable:,} | {parts}")
    assert total <= 20_000_000, "over the 20M parameter cap"
    ens = C.MixedEnsemble(threshold=thr)

    def run(split):
        batches = C.make_batches(C.encode(data[split], feats, labels, is_train=False), 256, shuffle=False)
        outs = [C.member_outputs(m, batches)[:2] for m in members]
        p, s = ens.combine([o[0] for o in outs], [o[1] for o in outs])
        return p, s, C.evaluate_outputs(p, s, data[split], labels)

    _, _, dev_res = run("dev")
    for k in ("intent_correct", "tp", "fp", "fn"):
        assert dev_res[k] == selected[k], f"reloaded ensemble disagrees with the selected dev result on {k}"
    print("Reloaded ensemble reproduces the selected dev result exactly.")
    test_p, test_s, test_res = run("test")
    epochs = "/".join(str(ck["epoch"]) for ck in cks)
    for split, r in (("Validation", dev_res), ("Test", test_res)):
        print(f"{args.name} | {split:<10} | intent accuracy {r['intent_acc']:.2f} | slot micro-F1 {r['slot_f1']:.2f} | "
              f"TP {r['tp']} FP {r['fp']} FN {r['fn']}")
    os.makedirs(os.path.join(C.ROOT, "outputs/ens"), exist_ok=True)
    preds = [{"id": e["id"], "utt": e["utt"], "gold_intent": e["intent"],
              "pred_intent": labels.intents.itos[int(i)],
              "gold_slots": sorted(e["gold_spans"]), "pred_slots": sorted(C.spans_to_text(s, e["tokens"], labels))}
             for e, i, s in zip(data["test"], test_p.argmax(1).tolist(), test_s)]
    json.dump({"members": args.members, "threshold": thr, "params": {"total": total, "trainable": trainable, **parts},
               "epochs": epochs, "dev": dev_res, "test": test_res, "test_predictions": preds},
              open(os.path.join(C.ROOT, "outputs/ens", f"{args.name}_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
