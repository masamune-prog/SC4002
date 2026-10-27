# Shared dilated CNN for joint intent and slot prediction

This is an **AI-assisted joint intent and slot experiment**, implemented and executed at the user's request. The agreed comparison uses the single seed-42 CNN as the **Stage 1 baseline** and the three-seed CNN ensemble as the **Stage 2 improvement**. The earlier draft's restriction limiting AI to clarification and suggestions has been removed, according to the updated assignment guidance supplied by the user.

Both systems have been implemented and trained. Stage 1 has recorded validation results; its separate test evaluation is still pending. Stage 2 has recorded validation and test results. This folder is an initial project snapshot, not yet the complete two-stage submission notebook.

The system predicts a 60-class intent and exact slot spans through one shared word-sequence encoder. It uses GloVe 6B 300d word vectors, a small character CNN, four non-causal residual dilated convolutions (192 channels; dilations 1, 2, 4, 8), a pooled intent head, and an intent-conditioned BIO-constrained CRF slot head. All non-embedding model weights start from random initialisation. The GloVe rows are restricted to the training vocabulary and fine-tuned.

Each member contains **2,307,402 total/trainable parameters**, including the word embeddings. The prespecified three-member ensemble contains **6,922,206 parameters**: members have separate learned embedding tables, so their full counts are added. Model-selection code checks the 20M cap again. The model is selected on validation data before test inference.

<!-- RESULTS_START -->
Results generated from the frozen evaluation artifacts:

| Split | Intent accuracy | Exact slot micro F1 |
|---|---:|---:|
| Validation | 87.70% | 82.85% |
| Test | 87.59% | 80.96% |

Selected candidate: **ensemble**. Total parameters: **6,922,206**; trainable: **6,922,206**. Selected epoch(s): **26 / 17 / 33**.

These are measured outputs from this experiment. Existing repository baselines were not retrained here.
<!-- RESULTS_END -->

## What to open

- `results.ipynb`: an executed evidence report that reads the completed run logs, configuration, validation selection, final results, and actual error examples. It does **not** claim training occurred inside the report notebook.
- `results/results.json`: unrounded final metrics, exact integer counts, environment, selected epochs, and parameter count.
- `results/results.csv`: percentages rounded to two decimal places.
- `results/selection.json`: validation-only candidate comparison and frozen checkpoint hashes.
- `results/model.pt`: portable selected system, created by the inference export command.
- `runs/cnn192_seed*/console.log`, `history.jsonl`, `history.csv`, `setup.json`, and `summary.json`: original training evidence, without fabricated output cells.
- `diagnostics/`: retained implementation checks, preprocessing/scorer verification, and a training-subset overfit diagnostic.

## Environment

Git tracks the source, configuration, executed report, and original training evidence. Downloaded data, embeddings, caches, and `.pt` checkpoint files are excluded. A fresh checkout can inspect the saved report immediately; new inference or model evaluation requires retraining below or obtaining the exported `model.pt` separately. Absolute paths in the saved run metadata describe the machine on which the original experiment ran.

Training was run locally on CPU using Python 3.12.0, PyTorch 2.12.0+cpu, and NumPy 2.4.4. The existing training interpreter on this machine is:

```text
C:\Users\samuel\anaconda3\envs\env\python.exe
```

The report uses a separate existing Python environment with matplotlib, pandas, nbformat, nbclient, and ipykernel. A new environment can install all requirements with:

```powershell
python -m pip install -r cnn_improvement/requirements.txt
```

Run the following commands from the **parent project directory**, not from inside `cnn_improvement`. Replace `python` with the desired environment's interpreter when multiple Python installations are present. The reproducibility target is the recorded software and hardware; different versions or devices may produce slightly different results.

## Download and verify permitted assets

```powershell
python cnn_improvement/download_assets.py
```

The downloader obtains the original MASSIVE 1.0 archive and extracts only `1.0/data/en-US.jsonl`. It checks the official split counts: 11,514 training, 2,033 validation, and 2,974 test records. No records are added, removed, or re-split.

GloVe 6B 300d is downloaded from [Stanford NLP's Hugging Face mirror](https://huggingface.co/stanfordnlp/glove). Only the selected compressed ZIP member is transferred, and its size and CRC are checked before recording the extracted file's SHA256. `asset_manifest.json` records exact provenance. The Stanford website can be selected explicitly with `--glove-source stanford`; its archive bytes differ slightly from the mirror's, so reproduce this run using the default mirror. No pretrained contextual language model is used.

## Run correctness checks

```powershell
python -m unittest discover -s cnn_improvement/tests -v
python -m cnn_improvement.sanity_check
```

The CRF is checked against brute-force enumeration, padding invariance is checked, both tasks are checked for gradients into the shared encoder, and exact slot-set scoring/preprocessing is tested. The sanity check deliberately overfits 64 training examples with random word vectors; these diagnostic scores are not held-out performance.

## Train the prespecified seeds

```powershell
python -m cnn_improvement.train train --config cnn_improvement/configs/cnn192.json --seed 42 --run-dir cnn_improvement/reproduction/runs/cnn192_seed42
python -m cnn_improvement.train train --config cnn_improvement/configs/cnn192.json --seed 43 --run-dir cnn_improvement/reproduction/runs/cnn192_seed43
python -m cnn_improvement.train train --config cnn_improvement/configs/cnn192.json --seed 44 --run-dir cnn_improvement/reproduction/runs/cnn192_seed44
```

The configuration allows up to 36 epochs, with validation-based early stopping. Training uses AdamW, warmup/cosine learning-rate decay, dropout, gradient clipping, and training-frequency-based word dropout. Both task losses train the same encoder. No external or AI-generated task examples are used. The vocabulary is fitted on training examples only. `annot_utt` supplies supervision; the model receives only features computed from `utt`.

These commands use fresh directories under `reproduction/`, preserving the committed evidence under `runs/`. Completed runs are protected from accidental overwriting. Use another new run directory for a subsequent experiment. To resume an interrupted run with its original settings, append `--resume` to its original training command.

## Select on validation, then evaluate once on test

```powershell
python -m cnn_improvement.train select --run-dirs cnn_improvement/reproduction/runs/cnn192_seed42 cnn_improvement/reproduction/runs/cnn192_seed43 cnn_improvement/reproduction/runs/cnn192_seed44 --output cnn_improvement/reproduction/results
python -m cnn_improvement.train final --output cnn_improvement/reproduction/results
python -m cnn_improvement.train final --output cnn_improvement/reproduction/results --verify
```

Selection compares the individual checkpoints and their full ensemble, using the mean of validation intent accuracy and exact slot micro F1. An exact tie favours fewer parameters. The final command verifies checkpoint hashes, reproduces validation counts, and then evaluates test data. `--verify` repeats the **already frozen** evaluation and checks exact integer-count agreement; it makes no new model selection.

For the recorded experiment, the system was frozen using validation results before test evaluation. No model tuning or candidate selection followed the test results. The report's error examples are post-freeze observations only.

Intent accuracy counts correct utterance labels. Slot F1 compares per-utterance sets of exact `(slot type, original span text)` pairs, pools TP/FP/FN, and computes `100 * 2TP / (2TP + FP + FN)`. It is not token accuracy. Decoding preserves original utterance span text.

## Export and run inference

```powershell
python -m cnn_improvement.predict --export --checkpoint cnn_improvement/reproduction/results/model.pt
python -m cnn_improvement.predict --checkpoint cnn_improvement/reproduction/results/model.pt --text "wake me up at five am this week"
```

The commands above use the newly reproduced run. On the original training machine, the already exported model is at `cnn_improvement/results/model.pt`; it is the default when `--checkpoint` is omitted. Inference requires only the utterance text and the selected exported model. No gold intent or slot annotation is supplied. The example above is the assignment's illustrative utterance; it is not an added training example.

## Generate the executed report

Run this with the report environment after final evaluation:

```powershell
python cnn_improvement/make_report.py
```

The script validates the recorded selection and epoch evidence, creates CSV results and learning curves, selects two actual mismatches from the saved predictions, updates the result table in this README, and executes `results.ipynb` with a Python kernel. All notebook outputs are produced by execution over the saved evidence. The original model-training logs remain separate and intact.

Summed training times are the sum of recorded member-process durations. If runs executed concurrently, this is not elapsed wall-clock time. Inference timings include batching and decoding on the recorded machine. They are not evidence of a speedup over a baseline without a matched benchmark.

## Source files

| File | Purpose |
|---|---|
| `model.py` | Shared residual dilated CNN, character encoder, joint heads, constrained CRF |
| `data.py` | Official splits, train-only vocabularies, encoded examples, embedding subset |
| `metrics.py` | Exact annotation parsing, original-text span recovery, shared scoring |
| `train.py` | Training, checkpoint reload verification, validation selection, final evaluation |
| `predict.py` | Export and utterance-only inference |
| `download_assets.py` | Verified public asset download/extraction |
| `make_report.py` | Executed report generated from actual saved artifacts |

Convolutional sequence labelling is motivated by [Strubell et al., 2017](https://aclanthology.org/D17-1283/). This implementation is an experiment for MASSIVE; the paper's accuracy or speedups are not assumed to transfer to this system. The [MASSIVE repository](https://github.com/alexa/massive) and [GloVe project](https://nlp.stanford.edu/projects/glove/) document the data and permitted pretrained word vectors.
