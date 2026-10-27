"""Build an executed evidence notebook from completed, frozen training runs.

This reports saved training/evaluation artifacts; it never trains a model or
chooses a model using test data. Run with the Python environment containing
matplotlib, pandas, nbformat, nbclient and ipykernel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_evidence(root, directory):
    results = read_json(directory / "results.json")
    selection = read_json(directory / "selection.json")
    if results["selection_sha256"] != sha256(directory / "selection.json"):
        raise ValueError("Results do not match the frozen selection manifest")
    if not results["reload_verified"]:
        raise ValueError("Checkpoint reload has not been verified")
    if results["parameters"]["total"] > 20_000_000:
        raise ValueError("Selected system exceeds the parameter limit")
    runs = []
    for saved in selection["run_dirs"]:
        path = Path(saved)
        if not path.exists():
            path = root / "runs" / Path(saved.replace("\\", "/")).name
        setup, summary = read_json(path / "setup.json"), read_json(path / "summary.json")
        history = read_jsonl(path / "history.jsonl")
        if not summary["reload_verified"] or summary["test_evaluated"]:
            raise ValueError("Run lacks verified validation-only model selection")
        if len(history) != summary["epochs_trained"]:
            raise ValueError(f"Incomplete epoch evidence in {path}")
        if [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
            raise ValueError(f"Repeated or missing epoch evidence in {path}")
        best = max(history, key=lambda row: row["dev_combined"])
        if best["epoch"] != summary["best_epoch"]:
            raise ValueError(f"Saved checkpoint is not the best logged dev epoch: {path}")
        runs.append({"path": path, "setup": setup, "summary": summary, "history": history})
    return results, selection, runs


def result_rows(results):
    rows = []
    for split in ("validation", "test"):
        metric = results[split]
        rows.append({
            "system": results["system"], "selected_candidate": results["selected_candidate"],
            "split": split, "total_parameters": results["parameters"]["total"],
            "trainable_parameters": results["parameters"]["trainable"],
            "intent_accuracy_percent": f"{metric['intent_acc']:.2f}",
            "exact_slot_micro_f1_percent": f"{metric['slot_f1']:.2f}",
            "combined_percent": f"{metric['combined']:.2f}",
            "best_epochs": "/".join(map(str, results["best_epochs"])),
            "epochs_trained": "/".join(map(str, results["epochs_trained"])),
            "examples": metric["n"], "correct_intents": metric["intent_correct"],
            "slot_tp": metric["tp"], "slot_fp": metric["fp"], "slot_fn": metric["fn"],
        })
    return rows


def plot_histories(runs, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.2, "figure.facecolor": "white"})
    colors = {"train": "#2365a0", "dev": "#cf6b22", "intent": "#2365a0",
              "slot": "#228258", "combined": "#7c4d9c"}

    def fill_axes(axes, run):
        history, summary = run["history"], run["summary"]
        epochs = [row["epoch"] for row in history]
        loss, metrics = axes
        for column, label, color in (("train_loss", "Training", colors["train"]),
                                     ("dev_loss", "Validation", colors["dev"])):
            loss.plot(epochs, [row[column] for row in history], label=label, color=color)
        loss.set(ylabel="Joint objective", xlabel="Epoch", title="Training and validation loss")
        loss.legend()
        for column, label, color in (("dev_intent_acc", "Intent accuracy", colors["intent"]),
                                     ("dev_slot_f1", "Exact slot micro F1", colors["slot"]),
                                     ("dev_combined", "Selection score", colors["combined"])):
            metrics.plot(epochs, [row[column] for row in history], label=label, color=color)
        metrics.axvline(summary["best_epoch"], color="#333333", linestyle="--", linewidth=1,
                       label=f"Selected epoch {summary['best_epoch']}")
        metrics.set(ylabel="Percent", xlabel="Epoch", title="Validation metrics")
        metrics.legend(fontsize=8, loc="lower right")

    output_paths = []
    for run in runs:
        seed = run["summary"]["seed"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        fill_axes(axes, run)
        fig.suptitle(f"Shared dilated CNN — seed {seed}")
        fig.tight_layout()
        path = directory / f"learning_curves_seed{seed}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 4 * len(runs)), squeeze=False)
    for row, run in zip(axes, runs):
        fill_axes(row, run)
        row[0].set_title(f"Seed {run['summary']['seed']}: losses")
        row[1].set_title(f"Seed {run['summary']['seed']}: validation")
    fig.suptitle("Completed training histories; checkpoint selection uses validation only", y=1.0)
    fig.tight_layout()
    path = directory / "learning_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def select_error_examples(directory):
    """Deterministic post-freeze examples, with observable differences only."""
    records = {split: read_jsonl(directory / f"{split}_predictions.jsonl")
               for split in ("dev", "test")}
    choices = []
    used = set()
    # Pick a slot mismatch and an intent mismatch in official record order.
    for split, kind in (("dev", "slot"), ("test", "intent")):
        candidates = [row for row in records[split]
                      if (row["gold_slots"] != row["predicted_slots"] if kind == "slot"
                          else row["gold_intent"] != row["predicted_intent"])]
        if candidates:
            row = candidates[0]
            choices.append({**row, "split": split,
                            "selection_reason": f"First {kind} error in original {split} order"})
            used.add((split, str(row["id"])))
    if len(choices) < 2:
        for split in ("dev", "test"):
            for row in records[split]:
                if (split, str(row["id"])) in used:
                    continue
                if row["gold_intent"] != row["predicted_intent"] or row["gold_slots"] != row["predicted_slots"]:
                    choices.append({**row, "split": split,
                                    "selection_reason": "First remaining mismatch in original split order"})
                    used.add((split, str(row["id"])))
                    if len(choices) == 2:
                        break
            if len(choices) == 2:
                break
    for row in choices:
        gold, predicted = set(map(tuple, row["gold_slots"])), set(map(tuple, row["predicted_slots"]))
        row["missing_slots"] = sorted(gold - predicted)
        row["unexpected_slots"] = sorted(predicted - gold)
        row["intent_correct"] = row["gold_intent"] == row["predicted_intent"]
        row["interpretation_limit"] = "Observed mismatch only; no causal diagnosis is established by this example."
    (directory / "error_examples.json").write_text(json.dumps(choices, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return choices


def update_readme(root, results):
    path = root / "README.md"
    if not path.exists():
        return
    lines = ["<!-- RESULTS_START -->", "Results generated from the frozen evaluation artifacts:", "",
             "| Split | Intent accuracy | Exact slot micro F1 |",
             "|---|---:|---:|"]
    for split in ("validation", "test"):
        score = results[split]
        lines.append(f"| {split.title()} | {score['intent_acc']:.2f}% | {score['slot_f1']:.2f}% |")
    lines += ["", f"Selected candidate: **{results['selected_candidate']}**. "
              f"Total parameters: **{results['parameters']['total']:,}**; "
              f"trainable: **{results['parameters']['trainable']:,}**. "
              f"Selected epoch(s): **{' / '.join(map(str, results['best_epochs']))}**.",
              "", "These are measured outputs from this experiment. Existing repository baselines were not retrained here.",
              "<!-- RESULTS_END -->"]
    content = path.read_text(encoding="utf-8")
    replacement = "\n".join(lines)
    content, count = re.subn(r"<!-- RESULTS_START -->.*?<!-- RESULTS_END -->",
                            lambda _: replacement, content, flags=re.S)
    if count != 1:
        raise ValueError("README must have exactly one results placeholder")
    path.write_text(content, encoding="utf-8")


def create_notebook(root, directory, output, execute=True):
    import nbformat as nbf
    from nbclient import NotebookClient

    relative_directory = os.path.relpath(directory, root).replace("\\", "/")
    cells = []
    md = lambda source: cells.append(nbf.v4.new_markdown_cell(source))
    code = lambda source: cells.append(nbf.v4.new_code_cell(source))
    md("# Shared dilated CNN: executed results and evidence\n\n"
       "This notebook **reads and checks completed training and evaluation artifacts**. "
       "Training ran separately through `python -m cnn_improvement.train train`; "
       "its original epoch logs remain in `runs/`. Executing this notebook does not retrain the model.\n\n"
       "**Disclosure:** AI-assisted implementation and execution. The agreed comparison uses the single seed-42 "
       "CNN as Stage 1 and the three-seed ensemble as Stage 2. Stage 1 has validation results; its separate "
       "test evaluation is still pending. This evidence report is not yet the complete two-stage submission notebook.")
    code("from pathlib import Path\nimport hashlib, json, sys\nimport pandas as pd\n"
         "from IPython.display import display, Image, Markdown\n\n"
         "ROOT = Path.cwd()\n"
         "if not (ROOT / 'model.py').exists():\n    ROOT = ROOT / 'cnn_improvement'\n"
         f"RESULTS = (ROOT / {relative_directory!r}).resolve()\n"
         "read_json = lambda path: json.loads(path.read_text(encoding='utf-8'))\n"
         "read_jsonl = lambda path: [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]\n"
         "results = read_json(RESULTS / 'results.json')\nselection = read_json(RESULTS / 'selection.json')\n"
         "assert hashlib.sha256((RESULTS / 'selection.json').read_bytes()).hexdigest() == results['selection_sha256']\n"
         "assert results['reload_verified'] and results['parameters']['total'] <= 20_000_000\n"
         "print('Report kernel:', sys.executable)\nprint('Training environment:')\n"
         "print(json.dumps(results['environment'], indent=2))\n"
         "print('Frozen selection:', selection['frozen_at_utc'])\n"
         "print('Final evaluation:', results['evaluated_at_utc'])")
    md("## Data provenance and configuration\n\n"
       "Only official MASSIVE 1.0 English training examples fit model parameters and vocabularies. "
       "The pretrained component is a GloVe word-vector table; the CNN, character encoder and output heads start from random weights. "
       "Both output heads share the word-sequence encoder. All parameters, including embeddings, count toward the cap.")
    code("manifest = read_json(ROOT / 'asset_manifest.json')\n"
         "print(json.dumps(manifest, indent=2))\n"
         "run_paths = []\n"
         "for saved_path in selection['run_dirs']:\n"
         "    path = Path(saved_path)\n"
         "    if not path.exists():\n        path = ROOT / 'runs' / saved_path.replace('\\\\', '/').split('/')[-1]\n"
         "    run_paths.append(path)\n"
         "setups = [read_json(path / 'setup.json') for path in run_paths]\n"
         "summaries = [read_json(path / 'summary.json') for path in run_paths]\n"
         "print('Recorded model/training configuration:')\n"
         "print(json.dumps(setups[0]['config'], indent=2))\n"
         "print('Training vocabulary:', setups[0]['vocab_words'], 'words;', setups[0]['vocab_chars'], 'characters')\n"
         "print('Receptive field:', setups[0]['receptive_field_tokens'], 'tokens')\n"
         "print('Maximum training utterance:', setups[0]['max_training_tokens'], 'tokens')\n"
         "print('Embedding coverage:', json.dumps(setups[0]['embedding_coverage'], indent=2))")
    md("## Correctness evidence\n\n"
       "The diagnostic subset scores below check implementation wiring; they are not validation or test performance. "
       "Gold-label round trips check preprocessing, not generalisation.")
    code("diagnostics = ROOT / 'diagnostics'\n"
         "preprocessing = read_json(diagnostics / 'preprocessing_validation.json')\n"
         "sanity = read_json(diagnostics / 'sanity.json')\n"
         "assert preprocessing['all_checks_passed'] and sanity['passed']\n"
         "print('Preprocessing checks:')\nprint(json.dumps(preprocessing, indent=2))\n"
         "print('Overfit sanity check:')\n"
         "print(json.dumps({key: sanity[key] for key in ['purpose', 'n_training_examples', 'steps', 'final', 'passed']}, indent=2))\n"
         "test_log = diagnostics / 'unit_tests.log'\n"
         "if test_log.exists():\n"
         "    raw = test_log.read_bytes()\n"
         "    encoding = 'utf-16' if raw.startswith((b'\\xff\\xfe', b'\\xfe\\xff')) else 'utf-8-sig'\n"
         "    print('Retained unit-test execution output:')\n    print(raw.decode(encoding, errors='replace'))\n"
         "else:\n    print('No retained unit-test log was supplied; no test-run claim is inferred.')")
    md("## Complete epoch histories\n\n"
       "Every row below is read from a saved training log. Checkpoints were selected using the mean of validation intent accuracy and exact slot micro F1. "
       "Each seed's checkpoint was reloaded and its validation counts reproduced.")
    code("for path, summary in zip(run_paths, summaries):\n"
         "    history = pd.DataFrame(read_jsonl(path / 'history.jsonl'))\n"
         "    assert len(history) == summary['epochs_trained']\n"
         "    best = history.loc[history['dev_combined'].idxmax()]\n"
         "    assert int(best['epoch']) == summary['best_epoch']\n"
         "    print(f\"Seed {summary['seed']}: selected epoch {summary['best_epoch']}; trained {summary['epochs_trained']} epochs\")\n"
         "    display(history[['epoch', 'train_loss', 'dev_loss', 'dev_intent_acc', 'dev_slot_f1', 'dev_combined', 'epoch_seconds', 'elapsed_seconds']])\n"
         "    display(Image(filename=str(RESULTS / f\"learning_curves_seed{summary['seed']}.png\")))")
    md("## Validation-only system selection\n\n"
       "The prespecified candidates are the individual seed checkpoints and their full ensemble. "
       "The largest validation mean wins; an exact tie favours fewer parameters. "
       "The selected checkpoint hashes were frozen before evaluating test data.")
    code("candidate_rows = []\n"
         "for candidate in selection['candidates']:\n"
         "    metric = candidate['validation']\n"
         "    candidate_rows.append({'candidate': candidate['name'], 'members': candidate['members'],\n"
         "        'total_parameters': candidate['parameters']['total'], 'dev_intent_accuracy': metric['intent_acc'],\n"
         "        'dev_exact_slot_f1': metric['slot_f1'], 'dev_combined': metric['combined']})\n"
         "display(pd.DataFrame(candidate_rows).round(2))\n"
         "print('Selected:', results['selected_candidate'])\n"
         "print('Selection rule:', selection['selection_rule'])")
    md("## Frozen validation and test results\n\n"
       "Metrics are percentages. Slot micro F1 pools exact `(slot type, original span text)` set matches across utterances. "
       "Inference timing below includes local batching and decoding; it is not a controlled comparison with another architecture.")
    code("table = pd.read_csv(RESULTS / 'results.csv')\n"
         "display(table[['split', 'total_parameters', 'trainable_parameters', 'intent_accuracy_percent',\n"
         "               'exact_slot_micro_f1_percent', 'combined_percent', 'best_epochs', 'epochs_trained']])\n"
         "for split in ['validation', 'test']:\n"
         "    metric = results[split]\n"
         "    assert abs(metric['intent_acc'] - 100 * metric['intent_correct'] / metric['n']) < 1e-10\n"
         "    denom = 2 * metric['tp'] + metric['fp'] + metric['fn']\n"
         "    assert abs(metric['slot_f1'] - 100 * 2 * metric['tp'] / denom) < 1e-10\n"
         "    print(split, json.dumps(metric, indent=2))\n"
         "print('Sum of selected members’ recorded training-process elapsed times (seconds):',\n"
         "      results['training_seconds_selected_members'])\n"
         "print('This sum is not overall wall-clock time when runs execute concurrently.')\n"
         "verification_path = RESULTS / 'verification.json'\n"
         "if verification_path.exists():\n"
         "    verification = read_json(verification_path)\n"
         "    assert verification['exact_counts_reproduced']\n"
         "    print('Independent frozen reload/evaluation reproduced all recorded integer counts.')")
    md("## Two observed errors after finalisation\n\n"
       "Examples are selected deterministically from saved predictions: the first validation slot mismatch and the first test intent mismatch, with a fallback if necessary. "
       "These are observations after model selection. They do not establish the cause of an error, and no tuning is performed in this notebook.")
    code("errors = read_json(RESULTS / 'error_examples.json')\n"
         "for i, example in enumerate(errors, 1):\n"
         "    print(f\"Example {i}: {example['split']} ID {example['id']}\")\n"
         "    print('Utterance:', example['utt'])\n"
         "    print('Gold intent:', example['gold_intent'])\n"
         "    print('Predicted intent:', example['predicted_intent'])\n"
         "    print('Gold slots:', example['gold_slots'])\n"
         "    print('Predicted slots:', example['predicted_slots'])\n"
         "    print('Missing slots:', example['missing_slots'])\n"
         "    print('Unexpected slots:', example['unexpected_slots'])\n"
         "    print('Selection:', example['selection_reason'])\n"
         "    print(example['interpretation_limit'], '\\n')")
    md("## Reproduce and inspect the implementation\n\n"
       "Run the commands in [README.md](README.md). The implementation is in "
       "[model.py](model.py), [data.py](data.py), [metrics.py](metrics.py), "
       "[train.py](train.py), and [predict.py](predict.py). "
       "The source files remain editable and the original execution logs remain under `runs/`.\n\n"
       "The experiment establishes measured performance for this implementation and configuration. "
       "It does not establish superiority over the existing repository baselines, whose training was not repeated under matched hardware and settings.")
    code("sources = []\n"
         "for filename in ['model.py', 'data.py', 'metrics.py', 'train.py', 'predict.py']:\n"
         "    path = ROOT / filename\n"
         "    sources.append({'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})\n"
         "display(pd.DataFrame(sources))\n"
         "print('Reproduction commands are in README.md; original runtime:', results['environment']['executable'])")
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"display_name": "Python 3 (report environment)",
                                      "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": sys.version.split()[0]}
    notebook.metadata["sc4002_report"] = {"purpose": "Executed report over saved experiment artifacts; not a retraining notebook",
                                          "results_sha256": sha256(directory / "results.json")}
    if execute:
        runtime = root / "cache" / "jupyter_runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(runtime))
        client = NotebookClient(notebook, timeout=120, kernel_name="python3",
                                resources={"metadata": {"path": str(root)}})
        notebook = client.execute()
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert all(cell.execution_count is not None for cell in code_cells)
        assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--no-execute", action="store_true", help="Generate a draft only; final delivery should execute it")
    args = parser.parse_args()
    root = args.root.resolve()
    directory = (args.results_dir or root / "results").resolve()
    results, selection, runs = validate_evidence(root, directory)
    rows = result_rows(results)
    with (directory / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_histories(runs, directory)
    select_error_examples(directory)
    update_readme(root, results)
    output = (args.notebook or root / "results.ipynb").resolve()
    notebook = create_notebook(root, directory, output, execute=not args.no_execute)
    print(f"{'Generated draft' if args.no_execute else 'Executed'} notebook: {output}")
    print(f"Code cells: {sum(cell.cell_type == 'code' for cell in notebook.cells)}")
    print(f"Results CSV and figures: {directory}")
    for row in rows:
        print(f"{row['split']}: intent {row['intent_accuracy_percent']}%, exact slot F1 {row['exact_slot_micro_f1_percent']}%")


if __name__ == "__main__":
    main()
