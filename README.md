Amazon MASSIVE Dataset Setup

## GloVe + CRF experiment

Download Stanford's [GloVe 6B archive](https://nlp.stanford.edu/projects/glove/)
and place it at `data/glove.6B.zip`. The word vocabulary is fitted on
`data/train.csv`; GloVe vectors are looked up from `utt` at both training and
evaluation time, including for words outside that vocabulary. No gold labels
enter the model inputs. The 100-dimensional word model has 3,053,994 parameters.

Submit `sbatch train_glove_crf.sbatch`, or run:

```bash
uv run python main_glove_crf.py --glove data/glove.6B.zip
uv run python eval_glove.py --glove data/glove.6B.zip
```

The best dev-selected checkpoint is saved under `checkpoints/glove_crf/`,
training history under `logs/glove_crf/`, and test CSV files under
`outputs/glove_crf/`.

## GloVe + BiGRU + CRF experiment

Run `sbatch train_bigru_glove_crf.sbatch` to train seeds 42–44 with the same
training split, GloVe vectors, loss, optimizer, and dev early stopping as the
GloVe Transformer. The two-layer bidirectional GRU has 3,455,914 parameters.
The three-seed mean dev slot F1 was 80.44%, versus 78.84% for the 3.05M
GloVe Transformer. The best-dev BiGRU checkpoint was seed 42; it scored
77.97% slot F1 and 82.82% intent accuracy on test.

`sbatch eval_bigru_glove_crf.sbatch` evaluates that checkpoint again.
Per-seed dev results, test metrics, and test predictions are under
`outputs/bigru_glove_crf/`.

## Diffusion training for BIO slots

`sbatch train_diffusion_glove.sbatch` trains a six-step discrete masked-label
diffusion head. The word encoder and intent head are the same GloVe + BiGRU
design; the slot head is a small Transformer denoiser. It predicts BIO tags
from partly masked gold tags during training and starts from fully masked tags
at inference. Its three-seed mean dev slot F1 was 69.90%. A one-step decode
was stronger than iterative decoding on the seed-44 checkpoint (77.58% versus
72.45% dev slot F1), so the iterative decoder was not retained as the final
model.

`sbatch train_diffusion_crf_glove.sbatch` trains a hybrid with masked-tag
denoising as an auxiliary slot objective and CRF sequence likelihood as the
main slot objective. Its three-seed mean dev slot F1 was 79.82%, versus
80.44% for the plain BiGRU + CRF. The best-dev hybrid checkpoint (seed 43)
scored 77.66% test slot F1 and 83.15% test intent accuracy. Re-evaluate it
with `sbatch eval_diffusion_crf_glove.sbatch`. The per-seed comparison and
test CSV files are under `outputs/diffusion_crf_glove/`.

Approach Vanilla Transformer

Using the massive dataset we first train our own bpe tokenizer on the training set of en-US.

The main model comprises of a single transformer encoder which takes in the tokenised sequence and outputs the predicted intent and slot tags.







final test: loss=1.7462  intent_accuracy=80.53%  slot_micro_f1=64.60%




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

