import json
from pathlib import Path

import pytest

from sc4002.eval import annot_to_slot_labels, bio_to_slot_labels, convert_to_bio, eval_preds

ANNOT = "wake me up at [time : nine am] on [date : friday]"


def test_annot_to_slot_labels():
    assert annot_to_slot_labels(ANNOT) == ["Other"] * 4 + ["time", "time", "Other", "date"]


def test_bio_to_slot_labels_round_trip():
    tags = ["O", "O", "O", "O", "B-time", "I-time", "O", "B-date"]
    assert bio_to_slot_labels(tags) == annot_to_slot_labels(ANNOT)
    assert convert_to_bio(bio_to_slot_labels(tags)) == tags


def test_eval_preds_perfect_and_partial():
    gold = annot_to_slot_labels(ANNOT)
    r = eval_preds(["alarm_set"], ["alarm_set"], [gold], [gold])
    assert (r["intent_acc"], r["slot_micro_f1"], r["ex_match_acc"]) == (1.0, 1.0, 1.0)
    pred = ["Other"] * 4 + ["time", "Other", "Other", "date"]          # 'nine' only: wrong boundary
    r = eval_preds(["alarm_query"], ["alarm_set"], [pred], [gold])
    assert r["intent_acc"] == 0.0 and r["ex_match_acc"] == 0.0
    assert r["slot_micro_f1"] == pytest.approx(0.5)                     # 1 of 2 predicted, 1 of 2 gold


DATA = Path(__file__).resolve().parents[1] / "1.0" / "data" / "en-US.jsonl"


@pytest.mark.skipif(not DATA.exists(), reason="run ./download_data.sh first")
def test_gold_labels_align_with_utt():
    for line in DATA.open(encoding="utf-8"):
        r = json.loads(line)
        assert len(annot_to_slot_labels(r["annot_utt"])) == len(r["utt"].split()), r["id"]
