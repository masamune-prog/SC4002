import itertools

import pytest
import torch

from sc4002.data import Encoder, Tokeniser, collate, load_splits, parse_annot
from sc4002.metrics import score, slot_micro_f1, tags_to_spans
from sc4002.model import CRF, JointTransformer
from sc4002.train import Config


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def test_split_sizes(splits):
    assert {k: len(v) for k, v in splits.items()} == {"train": 11514, "dev": 2033, "test": 2974}


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_gold_round_trip_is_perfect(splits, split):
    exs = splits[split]
    s = score([e.intent for e in exs], [e.intent for e in exs],
              [e.gold for e in exs], [tags_to_spans(e.words, e.tags) for e in exs])
    assert s["slot_f1"] == 100.0 and s["fp"] == 0 and s["fn"] == 0


def test_parse_annot():
    words, tags, gold = parse_annot("wake me up at [time : five am] [date : this week]")
    assert words == "wake me up at five am this week".split()
    assert tags == ["O", "O", "O", "O", "B-time", "I-time", "B-date", "I-date"]
    assert gold == [("time", "five am"), ("date", "this week")]


def test_tags_to_spans_edge_cases():
    w = "a b c d".split()
    assert tags_to_spans(w, ["I-x", "I-x", "O", "B-y"]) == [("x", "a b"), ("y", "d")]
    assert tags_to_spans(w, ["B-x", "I-y", "B-x", "I-x"]) == [("x", "a"), ("y", "b"), ("x", "c d")]


def test_slot_f1_multiset_and_empty():
    g = [[("x", "a"), ("x", "a")], []]
    assert slot_micro_f1(g, [[("x", "a")], []]) == pytest.approx(100 * 2 / 3)
    assert slot_micro_f1([[]], [[]]) == 100.0
    assert slot_micro_f1([[("x", "a")]], [[("y", "a")]]) == 0.0


def test_crf_matches_brute_force():
    torch.manual_seed(0)
    tags = ["O", "B-x", "I-x", "B-y", "I-y"]
    crf = CRF(tags)
    em = torch.randn(2, 4, len(tags))
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    T, S, E = crf.potentials()
    for b in range(2):
        n = int(mask[b].sum())
        scores = {}
        for path in itertools.product(range(len(tags)), repeat=n):
            s = S[path[0]] + em[b, 0, path[0]] + E[path[-1]]
            for t in range(1, n):
                s = s + T[path[t - 1], path[t]] + em[b, t, path[t]]
            scores[path] = s
        log_z = torch.logsumexp(torch.stack(list(scores.values())), 0)
        best = max(scores, key=lambda p: scores[p].item())
        gold = torch.tensor([list(best) + [0] * (4 - n)])
        nll = crf.nll(em[b : b + 1], gold, mask[b : b + 1])
        assert nll.item() == pytest.approx((log_z - scores[best]).item(), abs=1e-4)
        assert crf.viterbi(em[b : b + 1], mask[b : b + 1])[0] == list(best)


def test_crf_never_outputs_invalid_bio():
    torch.manual_seed(1)
    tags = ["O", "B-x", "I-x", "B-y", "I-y"]
    crf = CRF(tags)
    for path in crf.viterbi(torch.randn(50, 6, 5) * 5, torch.ones(50, 6, dtype=torch.bool)):
        names = [tags[i] for i in path]
        for prev, cur in zip(["O"] + names, names):
            if cur.startswith("I-"):
                assert prev[2:] == cur[2:] and prev != "O"


@pytest.mark.parametrize("tok", ["word", "clitic", "bpe"])
@pytest.mark.parametrize("pos", ["relative", "learned"])
def test_padding_invariance(splits, tok, pos):
    """Predictions for an utterance must not depend on what else is in the batch."""
    cfg = Config(tokeniser=tok, positions=pos, use_glove=False, d_model=64, heads=4, ffn=128, layers=2)
    train = splits["train"]
    enc = Encoder(train, Tokeniser(tok, train, 1000))
    model = JointTransformer(cfg, len(enc.units), len(enc.chars), enc.tags.itos, len(enc.intents)).eval()
    exs = [enc.encode(e) for e in splits["dev"][:8]]
    short = min(exs, key=lambda e: len(e.units))
    with torch.no_grad():
        alone = model(collate([short]))
        batched = model(collate(exs))
    i = exs.index(short)
    n = len(short.word_first)
    assert torch.allclose(alone[0][0], batched[0][i], atol=1e-4)
    assert torch.allclose(alone[1][0, :n], batched[1][i, :n], atol=1e-4)


def test_saved_bpe_is_reused(splits, tmp_path):
    """A BPE saved by one Tokeniser is loaded (not retrained) by the next, so runs agree."""
    from sc4002.data import Tokeniser

    train = splits["train"]
    first = Tokeniser("bpe", train, 500, save_dir=tmp_path)
    assert (tmp_path / "bpe_500.json").exists()
    second = Tokeniser("bpe", train, 500, save_dir=tmp_path)
    words = sorted({w for e in splits["dev"] for w in e.words})
    assert first.bpe.get_vocab() == second.bpe.get_vocab()
    assert [first.word_units(w) for w in words] == [second.word_units(w) for w in words]
