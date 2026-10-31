"""Compare three-seed GloVe Transformer and BiGRU dev results."""

import csv
import json
import statistics
from pathlib import Path


def main():
    output_dir = Path("outputs/bigru_glove_crf")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, parameters in (("transformer", 3_053_994), ("bigru", 3_455_914)):
        for seed in (42, 43, 44):
            if model == "transformer" and seed == 42:
                path = Path("logs/glove_crf/glove_crf_history.json")
            elif model == "transformer":
                path = Path(f"logs/glove_scale/baseline_seed{seed}/glove_crf_history.json")
            else:
                path = Path(f"logs/bigru_glove_crf/seed{seed}/glove_crf_history.json")
            best = max(json.loads(path.read_text(encoding="utf-8")),
                       key=lambda epoch: epoch["dev_slot_micro_f1"])
            rows.append(dict(
                model=model, seed=seed, parameters=parameters,
                best_epoch=best["epoch"],
                dev_slot_f1_pct=round(100 * best["dev_slot_micro_f1"], 4),
                dev_intent_accuracy_pct=round(100 * best["dev_intent_accuracy"], 4),
            ))
    with (output_dir / "dev_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for model in ("transformer", "bigru"):
        scores = [row["dev_slot_f1_pct"] for row in rows if row["model"] == model]
        summary.append(dict(
            model=model, parameters=next(row["parameters"] for row in rows
                                          if row["model"] == model),
            seeds=len(scores), mean_dev_slot_f1_pct=round(statistics.mean(scores), 4),
            sd_dev_slot_f1_pct=round(statistics.stdev(scores), 4),
        ))
    with (output_dir / "dev_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['model']}: {row['mean_dev_slot_f1_pct']:.2f}% "
              f"± {row['sd_dev_slot_f1_pct']:.2f}%")


if __name__ == "__main__":
    main()
