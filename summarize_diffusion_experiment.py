"""Compare three-seed GloVe BiGRU slot heads using dev histories."""

import csv
import json
import statistics
from pathlib import Path


def main():
    variants = (
        ("crf", "logs/bigru_glove_crf", "glove_crf_history.json", 3_455_914),
        ("diffusion", "logs/diffusion_glove", "diffusion_history.json", 4_643_498),
        ("diffusion_crf", "logs/diffusion_crf_glove", "diffusion_history.json", 4_643_498),
    )
    rows = []
    for name, directory, filename, parameters in variants:
        for seed in (42, 43, 44):
            path = Path(directory) / f"seed{seed}" / filename
            best = max(json.loads(path.read_text(encoding="utf-8")),
                       key=lambda epoch: epoch["dev_slot_micro_f1"])
            rows.append(dict(
                model=name, seed=seed, parameters=parameters,
                best_epoch=best["epoch"],
                dev_slot_f1_pct=round(100 * best["dev_slot_micro_f1"], 4),
                dev_intent_accuracy_pct=round(100 * best["dev_intent_accuracy"], 4),
            ))
    output_dir = Path("outputs/diffusion_crf_glove")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "dev_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name, _, _, parameters in variants:
        scores = [row["dev_slot_f1_pct"] for row in rows if row["model"] == name]
        summary.append(dict(
            model=name, parameters=parameters, seeds=len(scores),
            mean_dev_slot_f1_pct=round(statistics.mean(scores), 4),
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
