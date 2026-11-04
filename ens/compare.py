"""Dev-only comparison of member groups: mean intent / slot F1 / combined over seeds, and slot recall split by whether
the gold (type, value) occurs in the train split.   python ens/compare.py base=abl_nolex_s42,abl_nolex_s43 sm=..."""
import os, sys, statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import core as C


def main():
    data = C.load_data()
    labels = C.Labels(data["train"])
    seen = {(t, s) for e in data["train"] for t, s in e["gold_spans"]}
    for arg in sys.argv[1:]:
        name, tags = arg.split("=", 1)
        rows = []
        for tag in tags.split(","):
            o = torch.load(os.path.join(C.ROOT, "outputs/ens", f"{tag}_dev.pt"), weights_only=False)
            hit = {"seen": [0, 0], "unseen": [0, 0]}
            for e, sp in zip(data["dev"], o["spans"]):
                pred = C.spans_to_text(sp, e["tokens"], labels)
                for g in e["gold_spans"]:
                    k = "seen" if g in seen else "unseen"
                    hit[k][0] += g in pred
                    hit[k][1] += 1
            r = o["dev"]
            rows.append((r["intent_acc"], r["slot_f1"], r["combined"],
                         100 * hit["seen"][0] / hit["seen"][1], 100 * hit["unseen"][0] / hit["unseen"][1]))
        m = [st.mean(x[i] for x in rows) for i in range(5)]
        sd = st.stdev(x[2] for x in rows) if len(rows) > 1 else 0.0
        print(f"{name:<14} n={len(rows)} | intent {m[0]:.2f} | slot F1 {m[1]:.2f} | combined {m[2]:.2f} (sd {sd:.2f}) | "
              f"recall seen {m[3]:.1f} | recall unseen {m[4]:.1f}")


if __name__ == "__main__":
    main()
