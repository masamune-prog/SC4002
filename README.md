# NLP Stage-1 baseline: joint intent classification and slot filling (MASSIVE en-US)

## 1. Summary and status

A joint BiLSTM model that reads an English assistant request (`utt`) and predicts one of **60 intents** plus the
set of `[slot type : slot span]` pairs (**55 slot types**). It uses GloVe word vectors, character-CNN word features
and a CRF slot decoder, and has 4.88M parameters, all trained from scratch except the GloVe initialisation.

**Stage 1 is complete.** The model was frozen after dev-only ablations and a hyperparameter search, and test was
used once, for the final evaluation of that frozen checkpoint. No decision was made after seeing test.
Re-running `main.ipynb` repeats that same evaluation of the same checkpoint; it does not select anything.

## 2. Results

All numbers below are copied from the saved notebook outputs at commit `e128604`.

**Stage-1 system** (from `main.ipynb`, Section 9; the brief's report format):

| Model version | Split | Total # parameters | Total # trainable parameters | Intent accuracy (%) | Slot micro-F1 (%) | Best epoch (selected checkpoint) | Epochs trained |
|---|---|---|---|---|---|---|---|
| Stage-1 system | Validation | 4,879,850 | 4,879,850 | 86.97 | 81.54 | 26 | 40 |
| Stage-1 system | Test | 4,879,850 | 4,879,850 | 86.38 | 79.14 | 26 | 40 |

Combined score ((intent + slot F1) / 2): validation 84.25, test 82.76.

Raw test counts: 2,569 / 2,974 intents correct; slots TP 2,248, FP 618, FN 567.

**The same architecture over 3 seeds** (dev only, from `ablations.ipynb`; seeds 42 / 43 / 44, each at its
best-combined epoch, mean ± sample std):

| Dev intent acc | Dev slot F1 | Dev combined | Best-combined epoch (42 / 43 / 44) |
|---|---|---|---|
| 86.59 ± 0.33 | 81.63 ± 0.09 | 84.11 ± 0.13 | 26 / 21 / 30 |

The Stage-1 system is the seed-42 run. Its dev combined score is 84.25, the best of the three seeds (see Section 7).

## 3. How to reproduce

Tested with Python 3.11.9, PyTorch 2.6.0+cu124 (CUDA 12.4, cuDNN 9.1), on an NVIDIA GeForce RTX 4070 Ti (12 GB),
Windows 11.

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas matplotlib jinja2 jupyter    # tested: numpy 2.1.2, pandas 2.3.3, matplotlib 3.10.7
pip install nbformat nbclient                         # only for tools/ (headless build and run)
```

- Run `main.ipynb` top to bottom. It downloads MASSIVE 1.0 (official archive) into `data/` and GloVe 6B
  (about 820 MB, from Stanford) into `embeddings/` if they are missing.
- With the data already downloaded, a full run took **about 17 min** (1,035 s) on the RTX 4070 Ti.
- The three best-so-far checkpoints are written to `checkpoints/` (about 20 MB each).
- Headless: `python tools/run_nb.py main.ipynb` executes the notebook in a fresh kernel and saves it with outputs.
- Runs are deterministic on this setup: seeds are set, cuDNN runs in deterministic mode, and
  `CUBLAS_WORKSPACE_CONFIG` is set before torch is imported. The seed-42 run reproduced exactly (same per-epoch
  losses and scores) in all six runs: four of `main.ipynb`, plus one each in `ablations.ipynb` and
  `tuning.ipynb`. Other GPUs or library versions may give slightly different numbers.

## 4. Model

- **Inputs:** whitespace tokens of `utt` (the data is already lowercase) as word ids, plus character ids per word.
- **Word embeddings:** 300-d, initialised from GloVe 6B 300d for train-vocabulary words, fine-tuned.
  - Lookup is exact match, then the word with periods removed.
  - Words with no GloVe vector (7.2% of train types, mostly contractions and possessives like `what's`) get a
    random vector.
- **Char CNN:** 50-d character embeddings → Conv1d (100 filters, width 3) → max-pool → tanh.
- **Encoder:** [word ; char] → dropout 0.5 → 2-layer BiLSTM (256 per direction, dropout 0.5).
- **Intent head:** masked mean ⊕ max pooling → dropout → Linear 256 → ReLU → dropout → Linear 60.
  - Pooling happens before dropout. Dropout before max-pooling caused a train/eval mismatch, which the
    overfitting check caught.
- **Slot head:** dropout → Linear 111 (BIO tags) → hand-written linear-chain CRF with Viterbi decoding. The CRF
  was checked against brute-force enumeration.
- **Outputs:** an intent label, plus BIO tags turned into `(slot type, span text)` pairs.
- **Loss:** intent cross-entropy + CRF negative log-likelihood, per utterance, averaged over the batch.
- **Optimizer and training:** Adam, lr 1e-3, batch size 32, gradient clipping at 5.0, 40 epochs, no early stopping.
  Word dropout replaces words seen once in train with `<<UNK>>` at p = 0.5.
- `use_char` / `use_crf` / `freeze_emb` switches give the ablation variants (softmax slot head when the CRF is off).

## 5. Rules and evaluation

- **Data:** the official splits via the `partition` field, asserted as 11,514 / 2,033 / 2,974. The input is `utt`
  only; labels come from `intent` and `annot_utt`.
- **Train-only vocabularies:** words, characters, intents and tags are built from the train split. Dev and test
  words not seen in train map to `<<UNK>>`.
- **One shared evaluator** scores every experiment:
  - **Intent accuracy** = 100 × correct / total.
  - **Slot micro-F1:** per utterance, compare the *set* of `(slot type, span text)` pairs. Gold pairs are parsed
    directly from `annot_utt`. TP/FP/FN are pooled over all utterances, and F1 = 100 × 2TP / (2TP + FP + FN).
  - Turning tags into spans is part of the model: an orphan `I-x` tag (after `O` or a different type) starts a
    new span.
  - Unit tests cover hand-built cases, and gold tags round-trip to 100.00 F1 on every split.
- **Checkpoint selection:** the best dev combined score, (intent acc + slot F1) / 2, compared at full precision.
  Ties go to the earlier epoch. Best-intent and best-slot checkpoints are also kept; all three pick epoch 26.
  Each reloaded checkpoint reproduces its logged dev counts exactly.
- **Seed 42** was fixed before any results were seen.
- **Test was locked** in code until the model was frozen. Before the single test evaluation, the notebook asserts
  that the checkpoint is the one compared throughout: epoch 26 with dev 1,768 correct intents, TP 1,670, FP 415,
  FN 341.

## 6. Experiments (all dev only)

- **Ablations** (`ablations.ipynb`, `figures/ablation_dev_combined.png`): 5 variants × 3 seeds.
  - The CRF clearly helps slot F1.
  - The char CNN is within seed noise.
  - Frozen GloVe trades intent accuracy for slot F1.
  - The findings are summarised at the top of the notebook.
- **40-epoch trend check** (from the saved per-epoch logs, nothing retrained): the late-peaking ablation runs gain
  at most a few tenths of a point per 10 epochs. That is small next to the CRF gap, so the 40-epoch cap does not
  change the conclusions.
- **Tuning** (`tuning.ipynb`, `figures/tuning_sweep.png`):
  - One setting at a time around the Stage-1 config: hidden 128/384, layers 1/3, dropout 0.3/0.4, lr 5e-4/2e-3,
    batch 16/64, word dropout 0/0.25.
  - The rules were fixed in advance, with a **0.3 combined-point noise threshold** for combining settings,
    60-epoch reruns and the final pick. The top 3 then got a 3-seed check (4.49 GPU-hours, 21 runs).
  - **Outcome: the defaults were kept.** The best challenger (batch 16) was only +0.15 on the 3-seed mean.
  - lr 5e-4 and word dropout 0.25 had higher dev slot F1 on every seed, and batch 16 on two of three, but none
    had a higher combined score.

## 7. Limitations and notes for Stage 2

- **Seed luck:** seed 42 was the best of three seeds on dev. The Stage-1 test scores are therefore probably slightly
  above this architecture's average. (Hypothesis: not measured, because test was evaluated for seed 42 only.)
- **Dev → test slot-F1 drop:** slot F1 falls 2.40 points from dev to test, while intent accuracy falls 0.59.
  *Likely causes (unverified):*
  - Selection optimism: the epoch and config were chosen on dev.
  - Sampling variation between splits.
  - The test split's slightly higher out-of-vocabulary rate (3.51% vs 3.32% of tokens) may contribute.
- **Train/dev loss gap:** dev loss is lowest at epoch 11 and rises after that, while dev scores keep improving
  until about epoch 26. Likely reading: the model becomes over-confident on its errors.
- **Candidate Stage-2 directions** (untested):
  - A seed ensemble, to average out seed luck.
  - lr 5e-4 or word dropout 0.25, for slot F1.
  - Stronger or different regularisation (e.g. label smoothing, weight decay, embedding dropout) for the loss
    gap.
  - The parameter budget leaves about 15M of the 20M unused.

## 8. Repo layout

| Path | What it is |
|---|---|
| `main.ipynb` | The required Stage-1 pipeline: data checks, evaluator and tests, model, training, checkpoint reloads, single test evaluation, results table. |
| `ablations.ipynb` | Dev-only ablations (5 variants × 3 seeds) and their summary. Saves no checkpoints. |
| `tuning.ipynb` | Dev-only hyperparameter search and the seed check. Saves no checkpoints. |
| `tools/build_nb.py` | Generates all three notebooks from one source (shared cells are identical). `python tools/build_nb.py [main\|ablations\|tuning\|all] [out_dir]`. Rebuilding clears outputs. |
| `tools/run_nb.py` | Executes a notebook headlessly in a fresh kernel and saves it with outputs. |
| `figures/stage1_learning_curves.png` | Stage-1 train/dev loss and dev scores per epoch. |
| `figures/ablation_dev_combined.png` | Mean ± std dev combined per ablation variant. |
| `figures/tuning_sweep.png` | Seed-42 dev combined per tuning config, with the ±0.3 noise band. |

Git-ignored: `data/` (MASSIVE), `embeddings/` (GloVe), `checkpoints/` (model files). They are re-created by
running `main.ipynb`. `Group_Assignment.pdf` (the brief) is not tracked.
