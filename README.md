# Joint intent detection and slot filling on MASSIVE (en-US) with dilated convolutional networks

Branch `DanFeng-both` of the SC4002 group repository. Both stages of the assignment, built on a model family the rest
of the group is not using: **dilated convolutional networks** (the group already has BiLSTM and Transformer systems).

---

## 1. Executive summary

**Task.** For each English assistant request (`utt`), predict one of 60 intents and the set of `[slot type : slot span]`
pairs (55 slot types), with one joint model, at most 20M parameters, no pre-trained models other than word vectors,
trained on the official MASSIVE 1.0 en-US train split only.

**Approach.**

1. **Stage 1 (baseline):** a joint *iterated dilated CNN* (Strubell et al., 2017) with GloVe-initialised word embeddings,
   a character CNN, a pooled intent head and a CRF slot decoder. Width, dropout and learning rate were chosen on the
   validation split (`stage1_tuning.ipynb`).
2. **Stage 2 (AI-collaborative):** the frozen Stage 1 model was diagnosed on the validation split. Each weakness got a
   targeted change, and every change was tested by ablation on the validation split (`stage2_experiments.ipynb`).
   The final system is an ensemble of six small **gated dilated CNNs** that share one frozen GloVe + fastText table.

**Results** (intent accuracy / slot micro-F1, %; test evaluated once per stage, after freezing):

| System | Validation | Val. combined | Test | Test combined | Parameters |
|---|---|---|---|---|---|
| Joy, Stage 1 (two separate models) | 84.46 / 75.51 | 79.98 | 82.68 / 72.73 | 77.71 | 7.10M |
| Max, Stage 1 (BiLSTM-CRF) | 86.97 / 81.54 | 84.25 | 86.38 / 79.14 | 82.76 | 4.88M |
| **This branch, Stage 1 (ID-CNN-CRF)** | 86.82 / 82.12 | 84.47 | 85.57 / 79.31 | 82.44 | 3.47M |
| Joy, Stage 2 (3 x BiLSTM-CRF) | 88.93 / 83.17 | 86.05 | 87.22 / 81.84 | 84.53 | 17.92M |
| **This branch, Stage 2 (6 x gated dilated CNN)** | 88.59 / 83.65 | 86.12 | 87.32 / 81.64 | 84.48 | 18.22M |

Teammates' numbers are copied from their branches (`Max-Stage1` README, `Joy-both` notebook outputs).

**Reading the results honestly.**

* **Stage 1** is far ahead of Joy's Stage 1, and level with Max's BiLSTM-CRF while using fewer parameters
  (3.47M against 4.88M). On test, its slot F1 is higher (+0.17) and its
  intent accuracy is lower (-0.81).
* **Stage 2** improves on Stage 1 by +1.75 intent accuracy and +2.33 slot F1 on test.
  It is **statistically tied** with Joy's BiLSTM ensemble: +0.10 intent and -0.20
  slot F1 on test, with a similar parameter count. Single training runs of the same configuration differ by about 0.3
  combined points (Section 7.3), so neither system is clearly better.
* We did **not** re-tune or re-test after seeing the test scores. The test split was scored exactly once per stage.

---

## 2. How to reproduce

Tested with Python 3.13.9 (Anaconda), PyTorch 2.12.0 (CPU), numpy, pandas, matplotlib, nbformat and nbclient, on an
Apple M2 laptop (8 GB). No GPU is needed. If CUDA is available it is used.

```bash
pip install -r requirements.txt
python tools/run_notebook.py main.ipynb                 # about 100 min on the M2 CPU; downloads data and vectors if missing
python tools/run_notebook.py stage1_tuning.ipynb        # optional, about 3.5 h
python tools/run_notebook.py stage2_experiments.ipynb   # optional, about 3.3 h
```

* `main.ipynb` downloads MASSIVE 1.0 (official archive) into `data/`. It also downloads GloVe 6B 300d (about 820 MB
  zipped) and fastText `crawl-300d-2M` (about 1.5 GB zipped) into `embeddings/`. All three folders are git-ignored, as is
  `checkpoints/`.
* The notebooks are generated from `tools/core.py` (code) and `tools/notebook_text.py` (explanations) by
  `python tools/build_notebooks.py`. Rebuilding produces un-executed notebooks.
* Seeds are fixed. Runs are deterministic for a given machine *and thread count*. With a different number of CPU threads,
  floating-point sums are ordered differently and the training trajectories change slightly. Example: the six-member
  ensemble scored 86.37 validation combined with 2 threads (`stage2_experiments.ipynb`) and 86.12 with 3
  threads (`main.ipynb`).

---

## 3. Where to find each deliverable

| Requirement | Location |
|---|---|
| Load and verify the dataset, assert 11,514 / 2,033 / 2,974 | `main.ipynb` Section 2 |
| Preprocessing (BIO tags from `annot_utt`, vocabularies, tokenisation) | `main.ipynb` Sections 2.1, 4, 6.1, 7.2 |
| Shared evaluation functions with unit tests | `main.ipynb` Section 3 |
| Stage 1: architecture, inputs, outputs, loss, optimiser, parameter count within 20M | `main.ipynb` Section 6 |
| Stage 1: epoch logs, learning curves, selected checkpoint, reload check, test scores | `main.ipynb` Sections 6.4 to 6.6 |
| Stage 2: diagnosis, changes, parameter count, training logs, curves, checkpoint, test scores | `main.ipynb` Section 7 |
| Results table in the report format | `main.ipynb` Section 8, `results/results_table.csv` |
| Error analysis of the Stage 2 system | `main.ipynb` Section 9 |
| Stage 1 hyperparameter search (validation only) | `stage1_tuning.ipynb` |
| Stage 2 ensemble size, decoding options, ablations (validation only) | `stage2_experiments.ipynb` |
| AI chat history | `ai_collaboration.md` (submitted separately, not in this branch) |
| Two-page report | `report.pdf` (editable copy `report.docx`) |

---

## 4. Data, labels and evaluation

* **Splits.** The `partition` field of `1.0/data/en-US.jsonl`, with sizes asserted against the brief. Input is `utt` only.
  Nothing is re-split, added or removed.
* **Labels.** `annot_utt` is parsed into one BIO tag per whitespace token (111 tags), and the parser asserts that the
  recovered tokens are exactly `utt.split()`. The gold `(slot type, span text)` set is read directly from `annot_utt`.
* **Vocabularies** (words, characters, intents, tags) come from the train split only.
* **One scorer for everything.** Intent accuracy = 100 x correct / total. Slot micro-F1 compares *sets* of
  `(type, span)` pairs per utterance, pools TP, FP and FN over the split, and computes F1 = 100 x 2TP / (2TP + FP + FN).
  An `I-x` that does not continue an `x` span starts a new span. Hand-built unit tests pass, and gold tags score
  100.00 on all three splits.
* **Model selection** uses the validation combined score, (intent accuracy + slot F1) / 2, so a model cannot buy one
  metric with the other.
* **Test lock.** `check_split_allowed` raises an error if the test split is scored before that stage's checkpoint is
  frozen. Each stage's checkpoint is reloaded into a fresh model, and its validation TP, FP, FN and correct-intent counts
  are asserted equal to the logged best before test is scored.

---

## 5. Stage 1: joint iterated dilated CNN

| Component | Choice | Why |
|---|---|---|
| Word input | 300-d embeddings initialised from GloVe 6B and fine-tuned (96.55% of train words found), plus a 100-d character CNN | Word vectors carry meaning that 11.5k utterances cannot teach; characters cover unseen words |
| Rare words | Replaced by `<unk>` with probability 0.25 / (0.25 + count) during training | Teaches the `<unk>` vector to behave like a rare word, as unseen dev and test words will |
| Encoder | Projection to 384 channels, then a block of three width-3 convolutions (dilation 1, 2, 4) with ReLU, residual connection and layer norm, applied twice with shared weights | The receptive field is 29 tokens and every position is computed in parallel. Padding is zeroed after every layer so batch composition cannot change a prediction |
| Intent head | Masked mean + max pooling, MLP, 60 classes | Mean captures the overall topic, max captures single decisive words |
| Slot head | Linear emissions, linear-chain CRF, Viterbi decoding | The metric rewards only exact spans; a CRF scores whole tag sequences. Checked against brute-force enumeration |
| Loss and optimiser | Intent cross-entropy + CRF negative log-likelihood; AdamW (weight decay 0.01), batch 32, clipping 5, learning rate halved on a 3-epoch plateau, early stopping after 10, at most 50 epochs | Standard and stable |

**Tuning** (`stage1_tuning.ipynb`, validation only, rules fixed before running):

| Learning rate | Channels | Dropout | Val. intent | Val. slot F1 | Val. combined | Best epoch |
|---|---|---|---|---|---|---|
| 1e-03 | 384 | 0.3 | 87.01 | 82.64 | 84.83 | 45 |
| 1e-03 | 384 | 0.5 | 86.97 | 82.08 | 84.52 | 40 |
| 1e-03 | 256 | 0.5 | 86.87 | 81.79 | 84.33 | 42 |
| 1e-03 | 256 | 0.3 | 86.42 | 82.16 | 84.29 | 30 |
| 2e-03 | 256 | 0.3 | 86.08 | 81.47 | 83.78 | 41 |
| 2e-03 | 384 | 0.3 | 85.78 | 81.21 | 83.49 | 38 |
| 2e-03 | 256 | 0.5 | 85.24 | 79.77 | 82.51 | 24 |
| 2e-03 | 384 | 0.5 | 82.64 | 79.83 | 81.24 | 49 |

The two best settings were re-run with a second seed:

| Setting | Seed 42 | Seed 43 | Mean |
|---|---|---|---|
| lr 1e-03, 384 ch, dropout 0.3 | 84.83 | 83.93 | 84.38 |
| lr 1e-03, 384 ch, dropout 0.5 | 84.52 | 84.48 | 84.50 |

The setting with the higher mean was chosen (384 channels, dropout 0.5, learning rate 1e-3). Its final training in
`main.ipynb` selected epoch 45 of 50.

---

## 6. Stage 2: diagnosis, changes and final system

### 6.1 Diagnosis of the frozen Stage 1 model (validation split)

| Finding | Number |
|---|---|
| Intent accuracy, utterances **with** a word unseen in training (364) vs **without** (1669) | 77.20 vs 88.92 |
| Slot F1, same split | 69.43 vs 85.17 |
| Intent errors that pick an intent from a different scenario | 190 of 268 |
| Slot F1 when the intent is right vs wrong | 84.22 vs 63.26 |
| Utterances with an invalid BIO sequence | 9 |
| Utterances with a slot type never seen with the predicted intent in train | 67 |
| Validation loss: lowest at epoch 10 (2.172), then rises to 2.491 by epoch 50 while training loss falls to 0.569 | overfitting |

### 6.2 Changes and their measured effect

Each row removes one component from the final configuration (seed 42; validation split; `stage2_experiments.ipynb`).
A drop larger than about 0.3 combined points is larger than seed noise. The noise estimate comes from the six seeds of
the final configuration, whose combined scores range from 84.79 to 85.24.

| Variant | Intent | Slot F1 | Combined | Change vs final | Trainable params |
|---|---|---|---|---|---|
| final configuration (seed 42) | 88.00 | 82.48 | 85.24 | +0.00 | 2,448,770 |
| whitespace tokens (no clitic split) | 87.06 | 82.41 | 84.73 | -0.50 | 2,463,870 |
| no frozen pre-trained vectors | 85.49 | 80.94 | 83.22 | -2.02 | 2,333,570 |
| GloVe only (no fastText) | 87.26 | 81.75 | 84.51 | -0.73 | 2,391,170 |
| no hashed char n-grams | 87.70 | 82.32 | 85.01 | -0.23 | 2,189,120 |
| no scenario head | 87.51 | 82.81 | 85.16 | -0.08 | 2,445,296 |
| no intent-aware slots | 87.75 | 82.03 | 84.89 | -0.35 | 2,437,250 |
| no BIO constraints | 87.75 | 81.91 | 84.83 | -0.40 | 2,448,770 |
| no label smoothing | 87.31 | 81.71 | 84.51 | -0.73 | 2,448,770 |
| + self-attention layer | 87.70 | 82.47 | 85.09 | -0.15 | 2,745,794 |
| + R-Drop (alpha 1) | 88.05 | 81.59 | 84.82 | -0.42 | 2,448,770 |
| wider members (256 ch, dropout 0.3, more rows) | 87.21 | 82.15 | 84.68 | -0.55 | 3,889,042 |
| fine-tuned GloVe instead of frozen tables | 87.51 | 82.49 | 85.00 | -0.24 | 3,402,570 |

**What this shows.**

* The **frozen pre-trained vectors** are by far the largest contributor, followed by using **fastText as well as GloVe**,
  **label smoothing**, the **clitic-aware tokeniser** (`what's` becomes `what 's`, so the stem matches a pre-trained vector), the
  **BIO constraints** and the **intent-aware slots**.
* The **scenario head** and the **hashed character n-grams** are within noise. They are kept because they are cheap and
  did not hurt, but we do not claim they help.
* The alternatives (self-attention, R-Drop, wider members, fine-tuned instead of frozen vectors) were all worse and were
  not adopted.

### 6.3 Decoding options (six-member ensemble, validation)

| Decoding | Intent | Slot F1 | Combined |
|---|---|---|---|
| top-1 intents, slot-type mask off | 89.18 | 83.57 | 86.37 |
| top-1 intents, slot-type mask on | 89.18 | 83.53 | 86.36 |
| top-3 intents, slot-type mask on | 88.79 | 83.59 | 86.19 |
| kNN over train vectors, k=10, weight 0.2 (+13,264,128 stored values) | 89.13 | 83.57 | 86.35 |
| kNN over train vectors, k=20, weight 0.4 (+13,264,128 stored values) | 88.88 | 83.57 | 86.22 |
| kNN over train vectors, k=20, weight 0.6 (+13,264,128 stored values) | 88.83 | 83.57 | 86.20 |

Neither slot-type masking, nor joint top-3 intent re-ranking, nor kNN retrieval over training sentence vectors helped.
kNN would also add more than 13M stored values, breaking the 20M cap. All three were rejected.

### 6.4 Final Stage 2 system

* **One member:**
  * Input per unit: [frozen GloVe ; frozen fastText] (600-d rows for train units plus GloVe's 2,000 most frequent words,
    shared by all members), a 100-d trainable embedding, a character CNN, and hashed character 3-to-5-gram embeddings.
  * Encoder: a projection to 192 channels, then six residual gated (GLU) dilated convolutions with dilation 1, 2, 4, 8, 1, 2.
  * Intent: mean + max pooling into an intent layer plus a scenario layer, combined as log p(intent) + log p(scenario).
  * Slots: tokens read at their first unit, plus the expected intent embedding, then an MLP, then a BIO-constrained CRF.
  * Loss: intent cross-entropy with label smoothing 0.1 + 0.5 x scenario cross-entropy + CRF negative log-likelihood.
  * Training: AdamW, 1 warm-up epoch then cosine decay over 40 epochs, dropout 0.4. From epoch 30 the weight average
    (SWA) is also a checkpoint candidate. It was selected for 3 of 6 members, each time by at most 0.02 points, so it
    made no measurable difference.
* **Ensemble:** six members differing only in seed. Intent probabilities are averaged, and Viterbi runs on the averaged
  emission and transition scores.
* **Parameters:** 18,215,220 total (6 x 2,448,770 trainable + 3,522,600 frozen word-vector values).
* **Ensemble size on validation** (`stage2_experiments.ipynb`, combined score by number of members):
  1: 85.24 | 2: 85.80 | 3: 86.07 | 4: 86.22 | 5: 86.11 | 6: 86.37. The mean single member scores 85.03.

---

## 7. Advantages, disadvantages and limitations

### 7.1 Advantages

* **No recurrence:** every position is computed in parallel, and a full Stage 1 training run takes about 18 minutes on a
  laptop CPU.
* **Fewer parameters for Stage 1** than the BiLSTM baselines, at comparable accuracy.
* **Every Stage 2 component is backed by an ablation**, and the components that did not help are reported as such.
* **Leakage is ruled out by construction:** a strict test lock, plus reload checks that compare exact counts.

### 7.2 Disadvantages

* **Fixed context:** the receptive field is fixed (29 tokens in Stage 1; 37 in Stage 2). That covers every training
  utterance (at most 35 tokens), but the longest test utterance has 61 tokens.
* **Small Stage 2 margin:** the gain over Stage 1 comes mostly from ensembling and frozen vectors, and the ensemble is six
  times the inference cost of one member.
* **Scenario head adds nothing:** it did not reduce cross-scenario intent errors. On test, 288 of 377 intent errors still pick a
  different scenario.

### 7.3 Limitations of the experiments

* **Seed noise:** single-seed ablations have noise of about 0.3 combined points.
* **Thread-count sensitivity:** results change slightly with CPU thread count (Section 2).
* **One test evaluation per stage:** this is by design, but it means the test comparison with teammates rests on one run
  each.
* **Unseen words remain the largest weakness:** on test, utterances with a word unseen in training score
  75.23 intent and 66.42 slot F1, against 90.01 and 85.34 without.
  Only 14.9% of unseen validation units hit a frozen vector, because the vector table was kept small to fit six members
  under 20M.

### 7.4 Future work

* **Mixed-family ensemble:** combine the CNN members with the group's BiLSTM members. Mixing different variants gave the
  largest gains we measured.
* **Better vector coverage:** spend parameter budget on a larger vector table rather than on more members.
* **Stronger intent signal for indirect requests**, for example "pink is all we need" (light colour) or "send a
  giggle my way" (joke).

---

## 8. Repository layout

| Path | Contents |
|---|---|
| `main.ipynb` | The required notebook: both stages, with logs, curves, checkpoints, reload checks, test scores, results table and error analysis |
| `stage1_tuning.ipynb` | Stage 1 grid search and seed check (validation only) |
| `stage2_experiments.ipynb` | Stage 2 ensemble size, decoding options, ablations (validation only) |
| `tools/core.py` | All source code, split into notebook cells |
| `tools/notebook_text.py`, `tools/build_notebooks.py` | Notebook explanations and the builder |
| `tools/run_notebook.py` | Executes a notebook headlessly and saves its outputs |
| `results/` | JSON and CSV written by the notebooks (every number in this README comes from here) |
| `figures/` | Learning curves and the ensemble-size plot |
| `report.pdf`, `report.docx` | Two-page report |
