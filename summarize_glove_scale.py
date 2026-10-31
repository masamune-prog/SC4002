"""Summarize the three-seed GloVe model-size experiment from dev histories."""

import csv
import json
import statistics
from pathlib import Path


def main():
    output_dir = Path("outputs/glove_scale")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for size, parameters in (("baseline", 3_053_994), ("scaled", 6_938_154)):
        for seed in (42, 43, 44):
            history_path = (
                Path("logs/glove_crf/glove_crf_history.json")
                if size == "baseline" and seed == 42 else
                Path(f"logs/glove_scale/{size}_seed{seed}/glove_crf_history.json")
            )
            history = json.loads(history_path.read_text(encoding="utf-8"))
            best = max(history, key=lambda epoch: epoch["dev_slot_micro_f1"])
            rows.append({
                "model": size,
                "seed": seed,
                "parameters": parameters,
                "best_epoch": best["epoch"],
                "dev_slot_f1_pct": round(100 * best["dev_slot_micro_f1"], 4),
                "dev_intent_accuracy_pct": round(100 * best["dev_intent_accuracy"], 4),
                "dev_loss_at_best_epoch": round(best["dev_loss"], 4),
            })
    with (output_dir / "dev_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for size in ("baseline", "scaled"):
        scores = [row["dev_slot_f1_pct"] for row in rows if row["model"] == size]
        summary.append({
            "model": size,
            "parameters": next(row["parameters"] for row in rows if row["model"] == size),
            "seeds": len(scores),
            "mean_dev_slot_f1_pct": round(statistics.mean(scores), 4),
            "sd_dev_slot_f1_pct": round(statistics.stdev(scores), 4),
        })
    winner = max(summary, key=lambda row: row["mean_dev_slot_f1_pct"])["model"]
    for row in summary:
        row["selected_on_dev"] = row["model"] == winner
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    print(f"Selected on mean dev slot F1: {winner}")
    for row in summary:
        print(f"{row['model']}: {row['mean_dev_slot_f1_pct']:.2f}% "
              f"± {row['sd_dev_slot_f1_pct']:.2f}%")


if __name__ == "__main__":
    main()
