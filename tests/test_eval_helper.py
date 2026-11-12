import pytest

from sc4002.eval import EvalMetric, SpanDecoder, parse_slots

TRUE_INTENTS = ["alarm_set", "weather_query", "play_music"]
TRUE_ANNOTS = [
    "wake me up at [time : nine am] on [date : friday]",
    "what is the weather in [place_name : new york]",
    "play some music",
]


@pytest.fixture
def metric():
    return EvalMetric(TRUE_INTENTS, TRUE_ANNOTS)


def test_parse_slots():
    assert parse_slots("wake me up at [time : nine am] on [date : friday]") == [
        ("time", "nine am"),
        ("date", "friday"),
    ]


def test_parse_slots_value_with_colon_and_extra_spaces():
    assert parse_slots("remind me at [time :  5:30   PM ]") == [("time", "5:30 PM")]


def test_parse_slots_none():
    assert parse_slots("play some music") == []


def test_perfect_predictions(metric):
    assert metric.evaluate(TRUE_INTENTS, TRUE_ANNOTS) == {
        "intent_accuracy": 1.0,
        "slot_precision": 1.0,
        "slot_recall": 1.0,
        "slot_f1": 1.0,
        "exact_match": 1.0,
    }


def test_all_wrong(metric):
    pred_intents = ["a", "b", "c"]
    pred_annots = ["[x : y]", "[x : y]", "[x : y]"]
    assert metric.evaluate(pred_intents, pred_annots) == {
        "intent_accuracy": 0.0,
        "slot_precision": 0.0,
        "slot_recall": 0.0,
        "slot_f1": 0.0,
        "exact_match": 0.0,
    }


def test_partial_slots(metric):
    pred_annots = [
        "wake me up at [time : nine am] on friday",  # missed date
        "what is the weather in [place_name : new york]",
        "play [artist_name : some] music",  # spurious slot
    ]
    scores = metric.slot_scores(pred_annots)
    # tp=2, gold=3, pred=3
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["f1"] == pytest.approx(2 / 3)
    # Only the weather utterance is fully correct.
    assert metric.exact_match(TRUE_INTENTS, pred_annots) == pytest.approx(1 / 3)


def test_wrong_slot_type_is_not_a_match(metric):
    pred_annots = list(TRUE_ANNOTS)
    pred_annots[1] = "what is the weather in [city : new york]"
    assert metric.slot_scores(pred_annots)["recall"] == pytest.approx(2 / 3)


def test_intent_accuracy_partial(metric):
    assert metric.intent_accuracy(
        ["alarm_set", "wrong", "play_music"]
    ) == pytest.approx(2 / 3)


def test_correct_slots_wrong_intent_is_not_exact_match(metric):
    pred_intents = ["alarm_query", "weather_query", "play_music"]
    assert metric.exact_match(pred_intents, TRUE_ANNOTS) == pytest.approx(2 / 3)


def test_no_slots_anywhere():
    metric = EvalMetric(["a"], ["hello"])
    assert metric.slot_scores(["hello"]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_length_mismatch_raises(metric):
    with pytest.raises(ValueError):
        metric.intent_accuracy(["alarm_set"])
    with pytest.raises(ValueError):
        metric.slot_scores(TRUE_ANNOTS[:2])


def test_mismatched_gold_raises():
    with pytest.raises(ValueError):
        EvalMetric(["a", "b"], ["x"])


def test_decoder_and_evaluator_use_exact_spans():
    utterances = ["wake me up at five am this week", "play music"]
    tags = [
        ["O", "O", "O", "O", "B-time", "I-time", "B-date", "I-date"],
        ["O", "O"],
    ]
    slots = SpanDecoder().decode_batch(utterances, tags)
    assert slots == [[("time", "five am"), ("date", "this week")], []]
    metric = EvalMetric(
        ["alarm_set", "play_music"],
        ["wake me up at [time : five am] [date : this week]", "play music"],
    )
    assert metric.evaluate_percent(["alarm_set", "play_music"], slots) == {
        "intent_accuracy": 100.0,
        "slot_precision": 100.0,
        "slot_recall": 100.0,
        "slot_f1": 100.0,
        "exact_match": 100.0,
    }
    mismatched_case = [[("time", "Five am"), ("date", "this week")], []]
    assert metric.slot_scores(mismatched_case)["f1"] == pytest.approx(0.5)


def test_decoder_repairs_orphan_i_and_separates_adjacent_b_tags():
    decoded = SpanDecoder().decode(
        "five am tomorrow morning", ["I-time", "I-time", "B-date", "B-date"]
    )
    assert decoded == [("time", "five am"), ("date", "tomorrow"), ("date", "morning")]


def test_decoder_rejects_misaligned_or_invalid_tags():
    decoder = SpanDecoder()
    with pytest.raises(ValueError):
        decoder.decode("five am", ["B-time"])
    with pytest.raises(ValueError):
        decoder.decode("five", ["time"])
