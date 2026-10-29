Amazon MASSIVE Dataset Setup

Follow these steps to download, extract, and inspect the Amazon MASSIVE NLU dataset.

1. Download and Extract

Run the following commands in your terminal to download the tarball and extract its contents:

curl https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz --output amazon-massive-dataset-1.0.tar.gz
tar -xzvf amazon-massive-dataset-1.0.tar.gz


2. Directory Structure

Once extracted, the 1.0 directory will have the following structure:

1.0
├── LICENSE
└── data
    ├── af-ZA.jsonl
    ├── am-ET.jsonl
    ├── ar-SA.jsonl
    ...

---

# WL_branch: joint Transformer + frozen GloVe (Stage-1 system)

A joint intent-classification + slot-filling model for MASSIVE en-US, built as an alternative to
the BiLSTM / dilated-CNN systems on the other branches. One shared Transformer encoder feeds an
intent head and a slot head with a BIO-constrained CRF; most of the 20M-parameter budget is spent
on a large **frozen GloVe vocabulary**, aimed at the weakness every branch found: words that never
appear in training.

## Rules followed

- Official MASSIVE 1.0 en-US splits from `data/{train,dev,test}.csv` (made by `preprocess.sh`);
  sizes asserted as 11,514 / 2,033 / 2,974. No re-splitting, no extra data.
- The only input is `utt`; labels come from `intent` and `annot_utt`.
- All vocabularies (words, characters, labels, BPE) are built from **train only**. The extra GloVe
  rows are chosen by GloVe's own frequency ranking, never by looking at dev/test words.
- No pretrained models; the only pretrained component is the GloVe 6B 300d word-vector table.
- Every model is asserted to have **≤ 20,000,000 total parameters**, frozen GloVe rows included.
- All selection uses dev only. Test was scored exactly once, in `wl_final.ipynb`, on the final
  checkpoint after the reload check. No ablation or seed run touched test.

## Repository layout

| Path | Contents |
|---|---|
| `src/sc4002/data.py` | CSV loading, `annot_utt` → BIO tags, tokenisers (whitespace, clitic, BPE), vocab, batching |
| `src/sc4002/vectors.py` | GloVe streaming, budget-filling frozen table, back-off lookup for unseen words |
| `src/sc4002/model.py` | `JointTransformer`, relative-position Transformer blocks, char-CNN, BIO-constrained `CRF` |
| `src/sc4002/metrics.py` | Shared scoring: intent accuracy, slot micro F1 |
| `src/sc4002/train.py` | `Config`, `build`, `fit` (training + best-dev checkpointing), `load_checkpoint`, ablation CLI |
| `src/sc4002/experiments.py` | The default and 7 ablations (overrides of `Config` defaults) |
| `data/bpe_2000.json` | The BPE tokeniser for `tok_bpe2k`, trained once on train utterances and reused by every run |
| `tests/test_core.py` | 16 tests (see below) |
| `colab_run.ipynb` | Colab runner for the whole experiment: round 1 and round 2, ending with the final choice |
| `colab_run_executed.ipynb` | `colab_run.ipynb` as executed on Colab, outputs kept |
| `wl_final.ipynb` | Final run: selection tables, train the chosen config with checkpoint, reload + dev check, test once |
| `results/ablations.jsonl` | One line per run as written by the training CLI on Colab: config, best epoch, dev scores, parameter counts, per-epoch history |
| `results/final_run.json` | Final run log: config, best epoch, dev scores, per-epoch history, checkpoint path |
| `results/final_curves.png` | Final run learning curves |
| `results/final_scores.json` | Final checkpoint's dev and test scores (test scored once) |
| `results/final_test_predictions.jsonl` | Final checkpoint's test predictions, one line per utterance (for error analysis) |

Git-ignored: `embeddings/` (GloVe zip + cached tables), `checkpoints/` (the final checkpoint
`checkpoints/wl_final_default_s42.pt`, 80.0 MB), `wl_code.zip`.

## Setup and running

Local (Python 3.12, managed with `uv`):

```bash
uv sync
uv run pytest -q tests
mkdir -p embeddings && curl -L -o embeddings/glove.6B.zip \
  https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
uv run python -m sc4002.train --exp default --seeds 42      # one run; appends to results/ablations.jsonl
```

`nlp.stanford.edu` was unreachable when this was set up; `downloads.cs.stanford.edu` is Stanford's
official mirror of the same file (862,182,613 bytes).

The CLI skips any (experiment, seed) already in the results file, so interrupted grids resume.
Device: CUDA if available, else CPU (`SC4002_DEVICE=mps` to use Apple GPU). On an M3 Pro CPU one
epoch takes ~37 s, so the runs are done on Colab.

Colab (T4 GPU, ~15 s per epoch, ~10 min per 40-epoch run):

1. Build the code bundle locally: `zip -r wl_code.zip src tests data pyproject.toml -x '*__pycache__*'`
2. Upload it to `MyDrive/SC4002_WL/` on Google Drive.
3. Open `colab_run.ipynb` in Colab, choose a GPU runtime, run the cells.
   GloVe is downloaded once to `MyDrive/SC4002_WL/embeddings/`; results go to
   `MyDrive/SC4002_WL/results/ablations.jsonl` (environment variables `SC4002_EMB_DIR`,
   `SC4002_RESULTS`, `SC4002_CKPT` point the code at Drive).

## Architecture (default config)

```
utterance → whitespace words
  per word:  frozen GloVe 300d (38,190-row table; back-off for unseen words)
           ⊕ trainable word embedding 128d (train vocab 5,304; word dropout)
           ⊕ char-CNN 100d (32-d chars, width-3 conv, max-pool)
  → concat 528 → dropout 0.25 → Linear 528→384 + LayerNorm → dropout 0.2
  → 4 × pre-LN Transformer block (d 384, 6 heads, FFN 1536 GELU, dropout 0.2, attention dropout 0.1)
       with T5-style relative position bias (32 buckets, shared across layers) → LayerNorm
  ├─ intent head: attention pooling ⊕ max pooling (768) → 384 → GELU → 60 intents
  └─ slot head:  word state ⊕ expected-intent vector (softmax(intent) × 60×64 table)
                 → 384 → GELU → 111 BIO emissions → linear-chain CRF (invalid BIO moves forbidden) → Viterbi
```

Unseen-word back-off for GloVe: exact word → punctuation stripped → dots removed → average of
parts split at `- . / & _ '` → zero vector. Slot tags are word-level and are read at each word's
first unit, so scoring is identical for every tokeniser.

**Budget filling (`top_k = -1`).** The table holds GloVe vectors for every train word (or its
back-off parts), then GloVe's most frequent words in order until the model reaches the cap:
rows = (20,000,000 − non-table parameters) // 300. The non-table parameters are fixed by the config.

| Component | Parameters | Trainable |
|---|---|---|
| Frozen GloVe table (38,190 × 300) | 11,457,000 | no |
| Trainable word embedding (5,304 × 128) | 678,912 | yes |
| Char-CNN | 10,788 | yes |
| Input projection 528 → 384 + LayerNorm | 203,904 | yes |
| Relative position bias (32 × 6) | 192 | yes |
| 4 Transformer blocks (1,774,464 each) | 7,097,856 | yes |
| Final LayerNorm | 768 | yes |
| Intent head | 318,781 | yes |
| Intent context table (60 × 64) | 3,840 | yes |
| Slot MLP | 215,151 | yes |
| CRF | 12,543 | yes |
| **Total** | **19,999,735** | **8,542,735** |

**Training.** Loss = intent cross-entropy (label smoothing 0.1) + CRF negative log-likelihood per
word. AdamW, lr 5e-4, weight decay 0.01 on weight matrices only; 1-epoch linear warm-up then
cosine decay to 0; batch 32; gradient clipping 1.0; word dropout α = 0.25; 40 epochs. Dev is
scored after every epoch and the epoch with the best dev combined score is kept.

There is **no early stopping**, on purpose: the cosine schedule only reaches small learning rates
in the last ~10 epochs, where the best dev epoch usually falls, and stopping early would cut it
short. All runs use the same 40-epoch schedule, which keeps ablations comparable.

**GloVe coverage** of the 464 dev word tokens that never occur in train (computed locally with the
real GloVe file; this depends only on the table, not on training):

| Frozen table | Rows | Unseen dev tokens with a vector | All dev tokens with a vector |
|---|---|---|---|
| train words only (`topk_0`) | 4,968 | 4.53% | 96.73% |
| + 15k frequent words | 16,357 | 60.78% | 98.60% |
| filled to the cap (`default`) | 38,190 | 80.17% | 99.24% |
| filled, smaller encoder (d 256, 3 layers) | 55,101 | 84.91% | 99.40% |
| filled, wider encoder (d 512, 8 heads) | 18,342 | 64.66% | 98.73% |

## Metrics

- Intent accuracy = 100 × correct / utterances.
- Slot micro F1 = 100 × 2TP / (2TP + FP + FN), with TP/FP/FN pooled over all utterances on exact
  (slot type, span text) matches.
- Per utterance the pairs are compared as a **multiset**, so a span that occurs twice must be
  predicted twice. The other branches use sets; the difference is 1 span on dev (2,012 vs 2,011
  gold spans) and should be agreed on before `main.ipynb` is merged.
- Dev combined = (intent accuracy + slot F1) / 2, used for checkpoint and config selection.
- Gold tags round-trip to exactly 100.00 on all three splits (tested).

## Tests (`uv run pytest -q tests`, 16 pass)

Split sizes; gold round-trip = 100.00 on train/dev/test; `annot_utt` parsing; span decoding edge
cases; multiset F1 and empty cases; CRF NLL and Viterbi against brute-force enumeration; CRF never
outputs invalid BIO; padding invariance for every tokeniser × {relative, learned} positions; a saved
BPE tokeniser is reused, not retrained.

## Experiment log

All runs use the current code and are dev only. `colab_run.ipynb` runs the whole experiment in one
notebook and writes to `MyDrive/SC4002_WL/results/ablations.jsonl`:

1. **Round 1:** the default and the 7 ablations in `experiments.py`, seed 42. Each ablation changes
   exactly one setting from the default:

   | Ablation | Change from default | Question |
   |---|---|---|
   | `no_glove` | no GloVe table | Do pretrained vectors help at all? |
   | `topk_0` | GloVe for train words only | Is the large frozen vocabulary worth the budget? |
   | `tok_clitic` | clitic split (`what's` → `what` `'s`) | Does splitting contractions help? |
   | `tok_bpe2k` | BPE subwords, 2k vocabulary | Do subwords beat whole words? |
   | `no_char` | no char-CNN | Is the spelling feature needed? |
   | `dec_softmax` | per-word softmax instead of the CRF | Does the CRF help slot filling? |
   | `no_intent_cond` | intent not fed to the slot head | Does intent conditioning help? |

2. **Round 2:** the 3 best round-1 ablations (chosen from the round-1 table, not fixed in advance)
   and the default, seeds 42, 43, 44.
3. **Choice:** highest mean dev combined over the three seeds; the notebook warns if the winner's
   lead over the default is smaller than the seed spread.

16 runs on a Colab T4 (torch 2.11.0+cu130), 2.69 h of training in total, ~10 min per run. Every
number in the tables below comes from `ablations.jsonl` and is printed in
`colab_run_executed.ipynb` (round 1 table, per-seed round 2 table, 3-seed summary) and
again in section 1 of `wl_final.ipynb`.

**Round 1 results** (seed 42, dev, sorted by combined):

| Run | Best epoch | Intent acc | Slot F1 | Combined | Δ vs default | Total params | Trainable |
|---|---|---|---|---|---|---|---|
| `no_char` | 31 | 87.85 | 82.34 | 85.10 | +0.31 | 19,999,747 | 8,493,547 |
| `tok_clitic` | 32 | 87.70 | 82.33 | 85.01 | +0.23 | 19,999,735 | 8,523,535 |
| `default` | 37 | 87.65 | 81.91 | 84.78 | +0.00 | 19,999,735 | 8,542,735 |
| `no_intent_cond` | 29 | 87.31 | 81.40 | 84.36 | -0.43 | 19,999,819 | 8,514,319 |
| `tok_bpe2k` | 33 | 87.16 | 81.31 | 84.24 | -0.54 | 19,999,831 | 8,106,031 |
| `dec_softmax` | 32 | 88.19 | 78.65 | 83.42 | -1.36 | 19,999,792 | 8,530,192 |
| `topk_0` | 30 | 86.52 | 80.17 | 83.35 | -1.44 | 10,033,135 | 8,542,735 |
| `no_glove` | 39 | 86.08 | 80.46 | 83.27 | -1.51 | 8,427,535 | 8,427,535 |

**Round 2 results** (dev combined per seed, then 3-seed means, sorted by mean combined):

| Run | Seed 42 | Seed 43 | Seed 44 | Mean intent | Mean slot F1 | Mean combined | Std | Δ vs default |
|---|---|---|---|---|---|---|---|---|
| `default` | 84.78 | 84.70 | 85.02 | 87.64 | 82.03 | 84.83 | 0.17 | +0.00 |
| `no_char` | 85.10 | 84.60 | 84.52 | 87.28 | 82.20 | 84.74 | 0.31 | -0.10 |
| `no_intent_cond` | 84.36 | 84.82 | 84.98 | 87.75 | 81.69 | 84.72 | 0.32 | -0.11 |
| `tok_clitic` | 85.01 | 84.65 | 84.02 | 87.51 | 81.62 | 84.56 | 0.50 | -0.27 |

**Choice: the default configuration.** It has the highest 3-seed mean and the smallest spread.

What the experiment shows:
- **What matters.** The pretrained vectors (`no_glove` −1.51), the large frozen vocabulary
  (`topk_0` −1.44, same trainable size, only the table shrinks) and the CRF (`dec_softmax` −1.36:
  slot F1 −3.26, intent +0.54) each cost well over 1 point when removed.
- **Within noise.** The round-1 gains of `no_char` (+0.31) and `tok_clitic` (+0.23) did not hold
  over three seeds. Both end below the default, by less than the seed spread (std up to 0.50).
  Intent conditioning (`no_intent_cond` −0.11) is also within noise. A single-seed ranking is not
  reliable at this scale, which is why round 2 exists.
- **BPE.** `tok_bpe2k` scores −0.54 in round 1. A likely reason, not tested separately: subword
  pieces share their parent word's GloVe vector, so BPE adds little beyond what the char-CNN and
  the GloVe back-off already give.

### Final run (`wl_final.ipynb`)

The default configuration, seed 42, trained once on the local CPU (M3 Pro, torch 2.14.1,
5 threads; 25.5 min, ~38 s per epoch). The notebook was run top to bottom from the
terminal with `nbclient` (the same as Run All), outputs kept. All numbers below are printed in it.

- **Parameters:** 19,999,735 total (under the 20,000,000 cap), 8,542,735 trainable.
- **Checkpoint:** `checkpoints/wl_final_default_s42.pt` (80.0 MB), best dev epoch **39** of 40.
- **Reload check:** the model rebuilt from the checkpoint's stored config, with the saved weights
  loaded, gives exactly the logged best dev score (asserted in the notebook).
- **Test** was scored once, after the reload check, and was not used for any decision.

| Split | Intent acc | Slot F1 | Combined |
|---|---|---|---|
| Dev (best epoch) | 87.70 | 83.08 | 85.39 |
| **Test** | **87.09** | **81.01** | **84.05** |

Learning curves: `results/final_curves.png`. Train loss falls throughout. Dev loss is flat from
about epoch 10 on, but dev scores keep rising as the cosine schedule lowers the learning rate. The
best epochs are 36–39, which is why there is no early stopping.

This run's dev score (85.39) is higher than the Colab seed-42 default run (84.78). Same code,
config and seed, but CPU instead of GPU: the two runs follow different numerical paths, so they
are different samples. The three Colab seeds of the default ranged from 84.70 to 85.02, so this
run is at the high end of the expected range. Its dev score played no part in the choice, which
was fixed before it ran.

## Data notes and caveats

- **Results are never edited by hand.** `ablations.jsonl` lines are written only by the training
  CLI, one per finished run.
- **Executed notebook.** `colab_run_executed.ipynb` is the Colab session that ran every
  cell in order, with all outputs kept. Its 3-seed table and final choice (`Final configuration:
  default (mean dev combined 84.83, +0.00 vs default; largest seed std 0.50)`) match the tables
  above, which are printed from `ablations.jsonl`.
- **"Run all" runs exactly the experiment.** The notebook has no optional cells after the main
  flow, so running all cells starts only round 1 and round 2.
- **BPE reproducibility.** The `tokenizers` BPE trainer learns a different vocabulary in each new
  Python process (checked: three trainings, three vocabularies, also with parallelism disabled).
  The tokeniser is therefore trained once on train utterances, saved as `data/bpe_2000.json` and
  loaded by every run (checked: three processes, identical vocabulary and splits).
- **GPU non-determinism.** Re-running a configuration with the same seed on GPU can change scores
  in the last decimals; this is why the final choice is made on three seeds.
- **No checkpoints for ablations.** Ablation and seed runs log scores only (~80 MB per checkpoint
  would be wasted). Only the final run saves a checkpoint. Save → reload → identical dev score was
  verified locally with a 2-epoch smoke run.
- **GloVe only.** An early version also used fastText (crawl-300d-2M); it was replaced by GloVe 6B
  300d before any experiment.
