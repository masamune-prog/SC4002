import csv

import pytest

pytest.importorskip("tokenizers")

from sc4002.tokenizers import BpeTokenizer


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["utt", "partition", "locale"])
        writer.writeheader()
        writer.writerows(rows)


def test_train_save_load_and_word_alignment(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path, [
        {"utt": "hello world", "partition": "train", "locale": "en-US"},
        {"utt": "hello there", "partition": "train", "locale": "en-US"},
    ])
    tokenizer = BpeTokenizer.train_from_csv(csv_path, vocab_size=15, expected_rows=2)
    encoding = tokenizer.encode("hello world")
    assert len(encoding.ids) > 2  # BPE has split at least one original word.
    assert set(encoding.word_ids) == {0, 1}
    aligned = tokenizer.align_word_labels(encoding, [7, 8])
    assert tokenizer.collapse_subword_labels(encoding, aligned) == [7, 8]
    assert aligned.count(-100) == len(encoding.ids) - 2

    path = tmp_path / "bpe.json"
    tokenizer.save(path)
    loaded = BpeTokenizer.load(path)
    assert loaded.encode("hello world").ids == encoding.ids
    restored = BpeTokenizer.from_json(tokenizer.to_json())
    assert restored.encode("hello world").ids == encoding.ids
    assert loaded.pad_id == 0
    assert loaded.unk_id == 1


def test_rejects_non_training_rows(tmp_path):
    csv_path = tmp_path / "train.csv"
    write_csv(csv_path, [
        {"utt": "held out example", "partition": "dev", "locale": "en-US"},
    ])
    with pytest.raises(ValueError, match="not in the English training split"):
        BpeTokenizer.train_from_csv(csv_path, expected_rows=None)
