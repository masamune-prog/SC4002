"""Ablations. Each entry overrides fields of train.Config; one factor changes at a time
relative to DEFAULT (dev-only selection). With top_k=-1 (default) the frozen GloVe
table takes whatever budget the rest of the model leaves, so every run stays <= 20M.
Listed in run order (most informative first)."""

DEFAULT: dict = {}

EXPERIMENTS: dict[str, dict] = {
    "default": DEFAULT,
    # pretrained vectors
    "no_glove": {"use_glove": False},  # do pretrained vectors help at all?
    "topk_0": {"top_k": 0},  # GloVe for train words only: is the large frozen vocabulary worth it?
    # tokenisation
    "tok_clitic": {"tokeniser": "clitic"},  # what's -> what 's
    "tok_bpe2k": {"tokeniser": "bpe", "bpe_vocab": 2000},  # BPE subwords
    "no_char": {"char_filters": 0},  # is the char-CNN needed?
    # slot decoder / heads
    "dec_softmax": {"decoder": "softmax"},  # does the CRF help?
    "no_intent_cond": {"intent_cond": False},  # does feeding the intent into the slot head help?
}

CORE = list(EXPERIMENTS)
