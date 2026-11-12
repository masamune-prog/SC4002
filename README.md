# SC4002 — Joint intent classification and slot filling (MASSIVE en-US)

Everything we submit is in **`main.ipynb`**, run top to bottom. The same notebook trains and evaluates both systems:

| Stage | System | Where in `main.ipynb` |
|---|---|---|
| Stage 1 | Basic joint Transformer: GloVe-initialised word embeddings, 1–2 Transformer encoder layers, an intent head and a BIO slot head with BIO-constrained Viterbi decoding. Learning rate and depth are tuned on dev. | Section 5 |
| Stage 2 | Ensemble of six joint networks (2 × GLU-CNN + CRF, 2 × GLU-CNN + span head, BiLSTM + CRF, Transformer + CRF) over a frozen GloVe ⊕ fastText table. Intent probabilities are averaged and slots are decided by span voting. | Sections 6–8 |

Both stages are scored with the **official MASSIVE evaluation**. `src/sc4002/eval/massive_eval.py` holds a
verbatim copy (Apache-2.0) of `convert_to_bio` and `eval_preds` from
[alexa/massive](https://github.com/alexa/massive/blob/f966f21846043aabef9b0f974fa7970027f43738/src/massive/utils/training_utils.py),
plus MASSIVE's own gold-label parser from `scripts/create_hf_dataset.py`. It computes three metrics:

- **Intent accuracy**, with `sklearn.metrics.accuracy_score`.
- **Slot micro-F1**, with `seqeval.metrics.f1_score`. Every word gets a slot type (`Other` outside a slot), MASSIVE's
  `convert_to_bio` turns those into BIO tags, and a slot counts as correct only if its type and exact word boundaries
  match.
- **Exact-match accuracy**: the intent is correct and the whole slot sequence is identical.

Section 4 of the notebook wraps `eval_preds` in `score(...)`, and every dev and test number in the notebook comes
from that one function. The code is unit-tested in `tests/test_massive_eval.py`. That test also checks that the
gold labels line up word for word with `utt` on all 16,521 records.

## Results

These numbers come from the committed `main.ipynb` run (NVIDIA RTX 6000 Ada, torch 2.6.0+cu124, Python 3.12).

| System | Split | Total params | Trainable params | Intent accuracy | Slot micro-F1 | Epochs |
|---|---|---|---|---|---|---|
| Stage 1: joint Transformer | Validation | 1,924,939 | 1,924,939 | 84.41% | 75.79% | 40 (best: 34) |
| Stage 1: joint Transformer | Test | 1,924,939 | 1,924,939 | 83.22% | 72.85% | 40 (best: 34) |
| Stage 2: 6-member ensemble | Validation | 18,414,920 | 14,892,320 | 88.54% | 84.39% | 40 per member (best: 34/28/31/28/30/26) |
| Stage 2: 6-member ensemble | Test | 18,414,920 | 14,892,320 | **87.42%** | **82.23%** | 40 per member (best: 34/28/31/28/30/26) |

All scores come from the official MASSIVE metric (see above). Stage 1 tried learning rates 1e-3 and 5e-4 with 1
and 2 encoder layers. Dev picked lr 5e-4 with 2 layers, at epoch 34. Stage 2 picked a vote threshold of 0.6 on dev.
Test exact-match accuracy is 70.04% for Stage 2.

Data use: the train split fits every parameter and vocabulary. Dev selects epochs, the Stage-1 grid configuration
and the Stage-2 vote threshold. The test split is scored once per stage, after the checkpoint is frozen and
reloaded from disk. The only model input is `utt`.

## Repository layout

```
main.ipynb                  the submission notebook (Stage 1 + Stage 2, with outputs)
src/sc4002/eval/            evaluation code: massive_eval.py (official MASSIVE metrics, used by the notebook),
                            plus the group's earlier EvalMetric / SpanDecoder helpers
tests/                      unit tests for the evaluation code
download_data.sh            downloads MASSIVE en-US, GloVe 6B 300-d and fastText crawl 300-d
ens/run_notebook.py         runs main.ipynb top to bottom and saves all outputs into it
ens/run_notebook.sbatch     the same, as a Slurm job
pyproject.toml, uv.lock     pinned Python environment (Python 3.12, torch 2.6.0+cu124)
```

## How to reproduce

You need Linux with an NVIDIA GPU (CUDA 12.4 driver or newer), about 6 GB of free disk space, `git`, `curl` and
`unzip`.

### 1. Get the code and the Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv; skip if `uv --version` already works
git clone -b submission https://github.com/masamune-prog/SC4002.git SC4002
cd SC4002
uv sync                                              # creates .venv and installs torch 2.6.0+cu124 and src/sc4002
```

Check that the environment works by running the evaluation unit tests. 18 should pass. A 19th, the full-dataset
alignment check, is skipped until the data is downloaded in step 2, and passes after that.

```bash
uv run pytest -q tests
```

### 2. Download the data and word vectors

```bash
./download_data.sh
```

This downloads about 2.4 GB and creates the three files the notebook reads:

```
1.0/data/en-US.jsonl            MASSIVE 1.0, en-US only (16,521 records)
embeddings/glove.6B.300d.txt    GloVe 6B 300-d (Stage 1 and Stage 2)
embeddings/crawl-300d-2M.vec    fastText Common Crawl 300-d (Stage 2)
```

The script prints `en-US records: 16521 (expected 16521)` when it is done. It skips files that already exist,
so it is safe to run again. Use the official GloVe file from this script. A modified copy, such as one with an
extra `<unk>` line, changes the Stage-1 numbers.

### 3. Run the notebook

On a machine without Slurm, run this from the repository root:

```bash
uv run python ens/run_notebook.py main.ipynb
```

On a Slurm cluster, run this instead. It requests the `UGGPU-TC1` partition, so on another cluster change the
`#SBATCH --partition` line first.

```bash
sbatch ens/run_notebook.sbatch
```

Both commands execute `main.ipynb` from top to bottom and save every output back into the notebook.
The full run took about 51 minutes on one RTX 6000 Ada: Stage 1 takes about 8 minutes and each Stage-2 member about 7–9 minutes. That time was measured with two copies sharing the GPU, so one run alone should be faster. To pick a GPU on a multi-GPU machine, put `CUDA_VISIBLE_DEVICES=<id>` in front of the
command.

You can also open `main.ipynb` in Jupyter or VS Code with the `.venv` kernel and use "Run All". Start the kernel
from the repository root, because the notebook reads `1.0/` and `embeddings/` relative to the working directory.

The run writes:

- `checkpoints/stage1/joint_transformer_best.pt`: the selected Stage-1 checkpoint
- `checkpoints/stage2/<member>.pt` and `checkpoints/stage2/ensemble.json`: the six Stage-2 members and the vote threshold
- `logs/stage2/<member>.json`: per-epoch history for each Stage-2 member

### 4. Read the results

Open `main.ipynb`:

- Section 5: Stage-1 parameter count, tuning grid, learning curves, and the reloaded checkpoint's validation and test scores.
- Section 6: Stage-2 parameter count, checked against the 20M limit.
- Section 7: each Stage-2 member's per-epoch training and dev scores, and the vote-threshold sweep.
- Section 8: the reloaded Stage-2 ensemble's validation and test scores.
- Section 9: the results table for both stages, which should match the table above.
- Section 10: error analysis on the Stage-2 test predictions.

### Optional: quick code check (about 3 minutes)

```bash
uv run python ens/run_notebook.py main.ipynb --smoke smoke.ipynb
```

This runs a 2-epoch copy of the whole notebook (both stages) into `smoke.ipynb` and leaves `main.ipynb`
unchanged. The smoke copy scores dev in place of test, so it never touches the test split.

## Reproducibility

- **Same machine, same numbers.** The notebook seeds every random generator and turns on PyTorch's deterministic
  mode (`torch.use_deterministic_algorithms(True)`, deterministic cuDNN, `CUBLAS_WORKSPACE_CONFIG=:4096:8`).
  We ran the notebook twice from scratch on the same GPU. Every printed number matched in both runs: each
  epoch's losses and scores, each member's selected epoch, the threshold sweep, the validation and test scores,
  and the error analysis. The two Stage-1 checkpoints had identical weights.
- **Different GPU, slightly different numbers.** Different GPUs (and different CUDA, cuDNN or torch versions) run
  different floating-point kernels, so training follows a slightly different path. Expect scores within a few
  tenths of a point of the table above, not an exact match. The Stage-1 grid can also pick a different
  configuration, because its four configurations are within 1 point of each other on dev.
- **The notebook checks itself.** After training, each stage reloads its saved checkpoint from disk and asserts
  that the reloaded model gives exactly the dev score it was selected with, before it scores test.
- **Use the pinned environment.** `uv sync` installs the exact package versions in `uv.lock`. Do not upgrade
  torch if you want to match the numbers above.
