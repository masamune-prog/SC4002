"""Markdown text and cell order for each notebook. ("md", text) is a markdown cell; ("code", name) is a cell from core.py."""

SHARED_SETUP = [
    ("md", """
## 1. Setup

Everything runs from the repository root. Missing data and word vectors are downloaded on first use into
`data/` and `embeddings/` (both git-ignored). On a machine with CUDA the GPU is used; otherwise the CPU.
"""),
    ("code", "setup"),
    ("md", """
## 2. Load and verify the dataset

MASSIVE 1.0, `1.0/data/en-US.jsonl`, split with the existing `partition` field. The only model input is `utt`;
labels come from `intent` and `annot_utt`. Nothing is re-split, added or removed.
"""),
    ("code", "load_data"),
    ("md", """
### 2.1 Labels: from `annot_utt` to BIO tags

`annot_utt` marks each slot as `[type : words]`. We turn it into one tag per whitespace token: `B-type` on the first
word of a slot, `I-type` on the following words, `O` elsewhere. The parser asserts that the tokens it recovers are
exactly `utt.split()`, so labels and inputs can never drift apart. The gold set of `(slot type, span text)` pairs is
read straight from `annot_utt` and is what every model is scored against.
"""),
    ("code", "parse"),
    ("md", """
## 3. Shared evaluation functions

One scorer is used for every experiment in this repository:

* **Intent accuracy** = 100 x correct intents / utterances.
* **Slot micro-F1**: for each utterance, compare the *set* of predicted `(slot type, span text)` pairs with the gold
  set. A true positive needs both the type and the exact span. TP, FP and FN are pooled over all utterances, and
  F1 = 100 x 2TP / (2TP + FP + FN).
* Turning tags into spans is part of the model, not the scorer. An `I-x` that does not continue an `x` span starts a
  new span (the usual CoNLL convention), so an invalid tag sequence is still scored, never silently repaired.

The checks below test hand-built cases, and confirm that gold tags reproduce 100.00 on every split.
"""),
    ("code", "scorer"),
    ("md", """
**Test lock.** The test split can only be scored after a stage's checkpoint is frozen. `check_split_allowed` raises
an error otherwise, which makes accidental test-set tuning impossible rather than just discouraged.
"""),
    ("code", "test_lock"),
    ("md", """
## 4. Vocabularies, word vectors and batching

Intent, scenario, tag, word and character vocabularies are built from the **training split only**. Dev and test words
never seen in training map to `<unk>` (the character-level features still see their spelling).

Word vectors (GloVe, fastText) are the only pre-trained component, as the brief allows. `vector_keys` tries the exact
word, then simple normalisations (`p.m.` -> `pm`), then the average of the parts of a contraction or hyphenated word.
"""),
    ("code", "vocab"),
    ("code", "word_vectors"),
    ("code", "batching"),
    ("md", """
## 5. Linear-chain CRF (shared by both stages)

A per-token softmax picks each tag on its own, so it can output `O I-time`, which is not a valid span. A
linear-chain CRF scores the **whole tag sequence**: the sum of per-token emission scores plus a learned score for
every tag-to-tag transition. Training maximises the log-probability of the gold sequence (the forward algorithm sums
over all sequences); prediction uses Viterbi, the same dynamic programme as in HMM tagging, to find the best
sequence exactly. The implementation is checked against brute-force enumeration of every possible path.
"""),
    ("code", "crf"),
    ("md", """
### 5.1 Training loop and checkpoint selection

`fit` trains on the train split only. After each epoch it scores dev with the shared scorer, logs the train and dev
losses, dev scores, learning rate and time, and saves a checkpoint whenever the **dev combined score**
((intent accuracy + slot F1) / 2) improves. Combined is used because the brief weights both tasks and a model that
trades one for the other should not be rewarded.
"""),
    ("code", "training"),
]

STAGE1_MODEL = [
    ("code", "stage1_features"),
    ("code", "stage1_model"),
]

LAYOUTS = {}

LAYOUTS["stage1_tuning"] = [
    ("md", """
# Stage 1 tuning: joint dilated CNN (validation split only)

This notebook chooses the Stage 1 hyperparameters. It uses **only the train and validation (dev) splits**; the test
split is never loaded into a model here. The chosen configuration is then trained once more in `main.ipynb`, which
is the notebook that holds the required Stage 1 evidence.

**Rules fixed before running:**

1. Grid: learning rate {1e-3, 2e-3} x channels {256, 384} x dropout {0.3, 0.5}, seed 42. Everything else is the
   base configuration shown in Section 6.
2. The two best configurations by dev combined score are re-run with seed 43.
3. The configuration with the higher two-seed mean is chosen.
"""),
    *SHARED_SETUP,
    ("md", """
## 6. Stage 1 model and inputs

The model is described in full in `main.ipynb` Section 6. In short: GloVe-initialised word embeddings and a
character CNN feed an iterated dilated convolution block; the intent head pools over the sentence, and the slot
head feeds a CRF.
"""),
    *STAGE1_MODEL,
    ("code", "s1_inputs"),
    ("md", """
## 7. Grid search (seed 42)
"""),
    ("code", "s1_tuning_grid"),
    ("md", """
## 8. Seed check of the two best configurations
"""),
    ("code", "s1_tuning_seeds"),
]

LAYOUTS["main"] = [
    ("md", """
# SC4002 Group Assignment: joint intent detection and slot filling on MASSIVE (en-US)

Every system here reads an English assistant request (`utt`) and, in one joint model, outputs one of 60 intents and
the set of `[slot type : slot span]` pairs (55 slot types).

## Summary

| | Stage 1: baseline | Stage 2: AI-collaborative improvement |
|---|---|---|
| Model family | Joint **iterated dilated CNN** (ID-CNN) + CRF | Ensemble of joint **gated dilated CNNs** + constrained CRF |
| Word input | GloVe-initialised, fine-tuned word embeddings + char-CNN | Clitic-aware sub-tokens; frozen GloVe and fastText, small trainable embedding, char-CNN, hashed char n-grams |
| Intent head | mean + max pooling, MLP | same, plus a scenario head combined hierarchically |
| Slot head | linear emissions, CRF | intent-aware emissions, CRF with hard BIO constraints |
| Training | AdamW, plateau schedule, early stopping on dev | AdamW, warmup + cosine schedule, label smoothing, weight averaging (SWA) over late epochs |

**Why a convolutional family?** Other members of the group built BiLSTM and Transformer systems. A dilated CNN
reads the whole utterance in parallel: three stacked width-3 convolutions with dilation 1, 2 and 4 see 15 tokens, and
iterating the block twice sees 29, more than 95% of MASSIVE utterances (95th percentile: 13 tokens). With no
recurrence, all positions are computed in parallel, so the models train on a laptop CPU (per-epoch times are printed
in Sections 6.4 and 7.4); that is what kept a multi-member Stage 2 ensemble and the ablations affordable.

All scores are printed by the cells that compute them (Sections 6.6, 7.6 and 8). Nothing in this notebook is typed
by hand.

## Where to find each requirement

| Requirement (brief, Section 5) | Location |
|---|---|
| Load and verify the dataset; assert 11,514 / 2,033 / 2,974 | Section 2 |
| Preprocessing into model inputs and labels | Sections 2.1, 4, 6.1, 7.2 |
| Shared evaluation functions (intent accuracy, slot micro-F1) | Section 3 |
| Stage 1 model: architecture, inputs, outputs, loss, optimiser; parameter count within 20M | Sections 6, 6.3 |
| Stage 1 training logs, per-epoch losses and validation scores, elapsed time, learning curves; best checkpoint | Sections 6.4, 6.5 |
| Stage 1 final evaluation: reload, validation check, test scores | Section 6.6 |
| Stage 2 diagnosis of Stage 1 weaknesses (validation split) | Section 7.1 |
| Stage 2 model, parameter count within 20M | Sections 7.2, 7.3 |
| Stage 2 training logs, learning curves, best checkpoint | Sections 7.4, 7.5 |
| Stage 2 final evaluation: reload, validation check, test scores | Section 7.6 |
| Results table in the report format | Section 8 |
| Error analysis of the Stage 2 system | Section 9 |

Hyperparameter search for Stage 1 is in `stage1_tuning.ipynb`, and the Stage 2 ablations are in
`stage2_experiments.ipynb`. Both use the validation split only.
"""),
    *SHARED_SETUP,
    ("md", """
## 6. Stage 1: joint iterated dilated CNN baseline

**Inputs.** Each whitespace token is represented by (a) a 300-d word embedding initialised from GloVe 6B and
fine-tuned, with a train-only vocabulary, and (b) a 100-d character-CNN vector (width-3 convolution over 50-d
character embeddings, max-pooled), which gives unseen words a representation built from their spelling. During
training a word is replaced by `<unk>` with probability 0.25 / (0.25 + its training count), so the `<unk>` vector
learns to behave like a rare word.

**Encoder.** A linear projection to C channels, then an iterated dilated block (Strubell et al., 2017): three width-3
convolutions with dilation 1, 2 and 4, each followed by ReLU, dropout, a residual connection and layer norm. The same
block is applied twice (shared weights). Padding positions are zeroed after every layer, so a token's representation
never depends on how long the other utterances in its batch are.

**Outputs and heads.**
* Intent: masked mean and max pooling over tokens, then Linear, ReLU, dropout, Linear to 60 intent logits.
* Slots: a linear layer gives 111 BIO emission scores per token, decoded by the CRF of Section 5 (transitions learned
  freely; no hand-written constraints in Stage 1).

**Loss and optimiser.** Intent cross-entropy + CRF negative log-likelihood (mean over utterances). AdamW (weight decay
0.01), batch size 32, gradient clipping at 5, learning rate halved when the dev combined score has not improved for 3
epochs, early stopping after 10 epochs without improvement (at most 50 epochs). Checkpoint: best dev combined score.

**Why these choices.** Word vectors (week 9) give the model prior knowledge of word meaning that 11.5k utterances
cannot teach. The CRF (sequence labelling with Viterbi, week 5) scores whole tag sequences, which matters because the
metric only rewards exactly correct spans. Learning rate, width and dropout were chosen on the validation split in
`stage1_tuning.ipynb`.

### 6.1 Inputs and vocabulary
"""),
    *STAGE1_MODEL,
    ("code", "s1_inputs"),
    ("md", """
### 6.2 Configuration chosen in `stage1_tuning.ipynb`
"""),
    ("code", "s1_main_config"),
    ("md", """
### 6.3 Model and parameter count
"""),
    ("code", "s1_main_build"),
    ("md", """
### 6.4 Training (train split only) and checkpoint selection

Each line is one epoch: training loss, validation loss, validation intent accuracy, slot micro-F1, combined score and
elapsed time. `<- saved` marks a new best validation combined score.
"""),
    ("code", "s1_main_train"),
    ("md", """
### 6.5 Learning curves
"""),
    ("code", "s1_main_curves"),
    ("md", """
### 6.6 Final evaluation of the Stage 1 checkpoint

The saved checkpoint is reloaded into a fresh model. Its validation counts must equal the ones logged for the selected
epoch (asserted), and only then is the test split unlocked and scored, once.
"""),
    ("code", "s1_main_final"),
    ("md", """
## 7. Stage 2: AI-collaborative diagnosis and improvement

The AI collaboration is summarised in the report (chat history submitted separately as `ai_collaboration.md`). The process was:
**diagnose** the frozen Stage 1 model on the validation split, **propose** changes that target each weakness,
**test** each change on the validation split (`stage2_experiments.ipynb`), and keep only what helped.

### 7.1 Diagnosis of the Stage 1 model (validation split only)
"""),
    ("code", "diagnosis"),
    ("code", "s2_diagnose_stage1"),
    ("md", """
**What the diagnosis says, and the response to each finding**

| Finding on the validation split | Stage 2 response |
|---|---|
| Utterances containing a word unseen in training score far lower on both tasks (see the "With a word unseen in train" line) | Clitic-aware sub-tokens (`what's` -> `what 's`, `wyatt's` -> `wyatt 's`, `p.` -> `p`) so the stem matches a pre-trained vector; frozen GloVe and fastText rows for train words plus GloVe's most frequent words; hashed character n-grams |
| Most intent errors cross scenarios (for example `general_quirky -> qa_factoid`) | A scenario head (18 classes, labels derived from the intent label) whose log-probabilities are added to the intent head's |
| Slot F1 is much lower when the intent is wrong | Intent-aware slot emissions: the predicted intent distribution selects a learned intent embedding that is added to every token before the slot layer |
| Invalid BIO sequences (an `I-x` with no `B-x`) | Hard BIO constraints in the CRF |
| Training loss keeps falling while validation loss rises (Section 6.5) | Smaller, more regularised members (dropout 0.4, label smoothing 0.1), weight averaging over the last 10 epochs, and an ensemble of members that differ only in their random seed |

Changes that were tried and **not** kept (validation evidence in `stage2_experiments.ipynb`): a self-attention layer
on top of the convolutions, R-Drop consistency regularisation, joint re-ranking of the top-3 intents with slot-type
masking, kNN retrieval over training-set sentence vectors, fine-tuned (instead of frozen) word vectors, and wider
(256-channel) members.

### 7.2 Stage 2 tokenisation and inputs

A token is split into units for the encoder, but tags stay on the original whitespace tokens: the encoder output at each
token's **first** unit is used for its slot tag, so spans and scoring are exactly as in Stage 1.
"""),
    ("code", "stage2_tokenise"),
    ("code", "stage2_features"),
    ("code", "s2_inputs"),
    ("md", """
### 7.3 Stage 2 model and parameter count

**One member.** Units are represented by [frozen GloVe ; frozen fastText] (600-d, shared by all members) + a 100-d
trainable embedding (with frequency-based `<unk>` replacement) + a char-CNN + the mean of 50-d hashed character
3-to-5-gram embeddings (fastText-style, learned from train). A projection to 192 channels feeds six residual **gated**
dilated convolutions (dilation 1, 2, 4, 8, 1, 2): each computes `A * sigmoid(B)` from a width-3 convolution
(Dauphin et al., 2017), so every channel learns how much of its new evidence to let through.

**Heads.**
* Intent: mean + max pooling, Linear, ReLU, dropout. An intent layer (60) and a scenario layer (18) read the same
  vector; final intent score = log p(intent) + log p(scenario of that intent).
* Slots: tokens are read at their first unit, the intent distribution selects an intent embedding that is added to each
  token, then Linear, ReLU, dropout, Linear to 111 emissions, decoded by a CRF in which illegal BIO transitions are
  penalised by -10,000.

**Loss.** Intent cross-entropy with label smoothing 0.1 + 0.5 x scenario cross-entropy + CRF negative log-likelihood.

**Ensemble.** Members differ only in their random seed. Intent: average of member probabilities. Slots: Viterbi on the
averaged emission and transition scores, which is the tag path with the highest summed member score. All members share
one frozen word-vector table, which is counted once.
"""),
    ("code", "stage2_model"),
    ("code", "ensemble"),
    ("code", "s2_build"),
    ("md", """
### 7.4 Training the members (train split only) and selecting the checkpoint

Each member: AdamW (learning rate 1e-3, weight decay 0.01), one warm-up epoch then cosine decay over 40 epochs, batch
size 32, clipping at 5. From epoch 30 the running average of the weights (SWA) is also scored on validation after every
epoch; for each member the better of "single epoch" and "weight average" by validation combined score is kept.
"""),
    ("code", "s2_train"),
    ("code", "s2_select"),
    ("md", """
### 7.5 Learning curves
"""),
    ("code", "s2_curves"),
    ("md", """
### 7.6 Final evaluation of the Stage 2 checkpoint
"""),
    ("code", "s2_final"),
    ("md", """
## 8. Results in the report format
"""),
    ("code", "results_table"),
    ("md", """
## 9. Error analysis of the Stage 2 system (test split, after freezing)

The same diagnosis as Section 7.1, now on the frozen Stage 2 system, followed by individual error cases. Nothing here
feeds back into either model.
"""),
    ("code", "error_analysis"),
]

LAYOUTS["stage2_experiments"] = [
    ("md", """
# Stage 2 experiments (validation split only)

Evidence for the Stage 2 design in `main.ipynb`. Everything here is trained on the train split and scored on the
validation split; the test split is never used.

1. **Ensemble size**: six members of the final configuration (seeds 42 to 47), and the validation score of
   ensembles of the first 1 to 6 members.
2. **Decoding options** on the six-member ensemble: slot-type masking, joint top-3 intent re-ranking, kNN retrieval.
3. **Ablations**: remove one component of the final configuration at a time (seed 42), and try alternatives that were
   not adopted.

Single runs differ by a few tenths of a point between seeds (Section 1 shows how much), so a difference smaller than
about 0.3 combined points is treated as noise.
"""),
    *SHARED_SETUP,
    ("code", "stage1_features"),
    ("code", "stage1_model"),
    ("code", "stage2_tokenise"),
    ("code", "stage2_features"),
    ("code", "stage2_model"),
    ("code", "ensemble"),
    ("md", """
## 6. Shared inputs and the experiment runner

`run_stage2` trains one member with the final configuration plus the given changes and reports its validation scores,
including the best single-epoch score and the best weight-averaged (SWA) score.
"""),
    ("code", "s2x_setup"),
    ("md", """
## 7. Ensemble size
"""),
    ("code", "s2x_ensemble_size"),
    ("md", """
## 8. Decoding options (six-member ensemble)
"""),
    ("code", "s2x_decoding"),
    ("md", """
## 9. Ablations and alternatives (seed 42)

`delta_combined` is relative to the final configuration with the same seed.
"""),
    ("code", "s2x_ablations"),
]
