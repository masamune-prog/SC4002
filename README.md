# SC4002 — Stage-2 mixed-architecture ensemble (MASSIVE en-US)

One system predicts both the intent and the slot spans of an utterance. It is an
ensemble of six joint networks that share one frozen GloVe + fastText word-vector
table:

| Member | Encoder | Slot head | Seed |
|---|---|---|---|
| `glu_crf_s42`, `glu_crf_s43` | gated dilated CNN | BIO-constrained CRF | 42, 43 |
| `glu_span_s42`, `glu_span_s43` | gated dilated CNN | span classifier | 42, 43 |
| `lstm_crf_s42` | 2-layer BiLSTM | BIO-constrained CRF | 42 |
| `tf_crf_s42` | 4-layer pre-LN Transformer | BIO-constrained CRF | 42 |

The ensemble averages the members' intent probabilities. For slots, it keeps every
span that at least 60% of members predict. The 60% threshold was chosen on dev.

All code is in `main.ipynb`. The notebook covers data loading, preprocessing,
scoring, the model, training, ensemble selection, a single test evaluation, the
results table and error analysis.

## Results

These come from the committed `main.ipynb` run on a Tesla V100-32GB with
torch 2.6.0+cu124.

| Split | Intent accuracy | Slot micro-F1 | Total params | Trainable params |
|---|---|---|---|---|
| Validation | 88.59% | 84.29% | 18,414,920 | 14,892,320 |
| Test | 87.53% | 81.92% | 18,414,920 | 14,892,320 |

Each member trains for 40 epochs. The member epochs chosen on dev were
40/35/31/28/26/33.

## Reproduce on a fresh machine

You need Linux with an NVIDIA GPU (CUDA 12.4 driver or newer), about 6 GB of
free disk space, and `curl` and `unzip`.

### 1. Install uv and the Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # skip if uv is already installed
git clone -b ensemble_lexicon <repo-url> SC4002
cd SC4002
uv sync                                              # creates .venv with Python 3.12 and torch 2.6.0+cu124
```

### 2. Download the data and word vectors

```bash
./download_data.sh
```

This script downloads about 2.4 GB and creates the three files the notebook
reads:

```
1.0/data/en-US.jsonl            MASSIVE 1.0, en-US only (16,521 records)
embeddings/glove.6B.300d.txt    GloVe 6B 300-d
embeddings/crawl-300d-2M.vec    fastText Common Crawl 300-d
```

Splits come from the dataset's own `partition` field: 11,514 train, 2,033 dev and
2,974 test examples.

### 3. Train the ensemble and evaluate it

On a Slurm cluster, run this from the repository root:

```bash
sbatch ens/run_notebook.sbatch
```

`ens/run_notebook.sbatch` requests the `UGGPU-TC1` partition. On another
cluster, change `#SBATCH --partition` to your GPU partition first.

On a machine without Slurm, run:

```bash
.venv/bin/python ens/run_notebook.py main.ipynb
```

Both commands execute `main.ipynb` from top to bottom and save every output back
into the notebook. Training writes these files:

- `checkpoints/stage2/<member>.pt`: the six member checkpoints
- `checkpoints/stage2/ensemble.json`: the selected members and vote threshold
- `logs/stage2/`: per-epoch history
- `logs/sc4002-notebook-<jobid>.out`: the Slurm log

Each member trains for about 5 minutes on a V100 and about 3 minutes on an
RTX 6000 Ada. The whole notebook took 19 minutes on the RTX 6000 Ada, well under
the 3-hour job limit. To see the results, open `main.ipynb`:

- Section 7: the dev scores of each member and the vote-threshold sweep.
- Section 8: the reloaded ensemble's validation and test scores.
- Section 9: the results table.

### Optional: quick code check

This command runs a 2-epoch copy of the notebook, writes it to `smoke.ipynb`,
and leaves `main.ipynb` unchanged:

```bash
.venv/bin/python ens/run_notebook.py main.ipynb --smoke smoke.ipynb
```

The smoke copy scores dev in place of test, so it never touches the test split.

## Notes on reproducibility

- Members are seeded, but cuDNN and CUDA kernels are not fully deterministic.
  On a different GPU, expect results within a few tenths of a point of the table
  above, not an exact match. A fresh clone following these steps on an RTX 6000
  Ada scored 88.39% / 84.30% on validation and 87.56% / 82.22% on test (intent
  accuracy / slot micro-F1), with the same 0.6 vote threshold.
- Data use: the train split fits every parameter and vocabulary. Dev selects
  each member's epoch (raw or SWA weights) and the vote threshold. The test split
  is scored once, in Section 8, after everything is frozen. The only model input
  is `utt`.
- Section 5 (Stage 1) is intentionally left for the group's own non-AI baseline.
  If `STAGE1` is not defined, the results table shows Stage 2 only.
