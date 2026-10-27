"""Semantic checks for alignment, exact scoring, and training-only vocabulary.

These synthetic fixtures are unit tests only; they are never model training
examples and never enter any official dataset split.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cnn_improvement.data import (
    MassiveDataset,
    build_vocab,
    collate_batch,
    encode_record,
    load_embedding_matrix,
    load_splits,
)
from cnn_improvement.metrics import (
    gold_bio_tags,
    parse_annotated_utterance,
    score_pair_predictions,
    score_predictions,
    tags_to_pairs,
)


def record(identifier, utt, annotated, intent="send", partition="train"):
    return {
        "id": identifier, "utt": utt, "annot_utt": annotated,
        "intent": intent, "partition": partition, "locale": "en-US",
    }


class SpanAlignmentTests(unittest.TestCase):
    def test_internal_whitespace_and_punctuation_preserved(self):
        utt = "email  Ana-Maria\tabout lunch  tomorrow"
        annotated = "email  [person : Ana-Maria]\tabout [event : lunch  tomorrow]"
        parsed = parse_annotated_utterance(annotated, utt)
        self.assertEqual(parsed["gold_pairs"], {("person", "Ana-Maria"), ("event", "lunch  tomorrow")})
        tags = gold_bio_tags(utt, annotated)
        self.assertEqual(tags, ["O", "B-person", "O", "B-event", "I-event"])
        self.assertEqual(tags_to_pairs(utt, tags), parsed["gold_pairs"])

    def test_adjacent_same_type_slots_stay_separate(self):
        utt = "tell alex sam"
        annotated = "tell [person : alex] [person : sam]"
        tags = gold_bio_tags(utt, annotated)
        self.assertEqual(tags, ["O", "B-person", "B-person"])
        self.assertEqual(tags_to_pairs(utt, tags), {("person", "alex"), ("person", "sam")})

    def test_repeated_pairs_use_assignment_set_semantics(self):
        utt = "tell alex about alex"
        annotated = "tell [person : alex] about [person : alex]"
        pairs = parse_annotated_utterance(annotated, utt)["gold_pairs"]
        self.assertEqual(pairs, {("person", "alex")})
        result = score_pair_predictions(["send"], ["send"], [pairs], [pairs])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 0))

    def test_mismatch_or_unrepresentable_annotation_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_annotated_utterance("hello [person : alex]", "hello sam")
        with self.assertRaises(ValueError):
            gold_bio_tags("helloalex", "hello[person : alex]")
        with self.assertRaises(ValueError):
            parse_annotated_utterance("hello [broken")
        with self.assertRaises(ValueError):
            tags_to_pairs("hello alex", ["O"])

    def test_orphan_i_recovery_and_no_slots(self):
        self.assertEqual(
            tags_to_pairs("five am monday", ["I-time", "I-time", "I-date"]),
            {("time", "five am"), ("date", "monday")},
        )
        self.assertEqual(tags_to_pairs("hello there", ["O", "O"]), set())
        result = score_pair_predictions(["greet"], ["greet"], [set()], [set()])
        self.assertEqual(result["slot_f1"], 100.0)


class ExactScoringTests(unittest.TestCase):
    def test_wrong_type_or_boundary_counts_as_both_fp_and_fn(self):
        gold = [{("time", "five am"), ("date", "monday")}]
        for predicted in [
            {("date", "five am"), ("date", "monday")},
            {("time", "five"), ("date", "monday")},
        ]:
            result = score_pair_predictions(["alarm"], ["alarm"], gold, [predicted])
            self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
            self.assertEqual(result["slot_f1"], 50.0)

    def test_pooled_micro_f1_is_not_mean_per_example_f1(self):
        result = score_pair_predictions(
            ["alarm", "send"], ["alarm", "alarm"],
            [{("time", "five am")}, {("person", "alex"), ("event", "lunch")}],
            [{("time", "five am"), ("person", "me")}, set()],
        )
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 2))
        self.assertEqual(result["intent_acc"], 50.0)
        self.assertEqual(result["slot_f1"], 40.0)
        self.assertEqual(result["combined"], 45.0)

    def test_raw_records_score_strings_without_vocab(self):
        rows = [record("0", "tell alex", "tell [person : alex]")]
        result = score_predictions(rows, ["send"], [["O", "B-person"]])
        self.assertEqual((result["intent_acc"], result["slot_f1"]), (100.0, 100.0))


class VocabularyAndBatchTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            record("0", "tell alex", "tell [person : alex]"),
            record("1", "tell alex tomorrow", "tell [person : alex] [date : tomorrow]"),
        ]
        self.vocab = build_vocab(self.train, verify_labels=False)

    def test_dev_words_never_enter_vocab_and_labels_are_not_inputs(self):
        before = json.dumps(self.vocab, sort_keys=True)
        dev = record("2", "tell zelda", "tell [person : zelda]", partition="dev")
        encoded = encode_record(dev, self.vocab)
        self.assertEqual(encoded["word_ids"][-1], self.vocab["unk_id"])
        self.assertNotIn("zelda", self.vocab["word_to_id"])
        self.assertEqual(json.dumps(self.vocab, sort_keys=True), before)
        changed_supervision = dict(dev, annot_utt="tell zelda")
        encoded2 = encode_record(changed_supervision, self.vocab)
        self.assertEqual(encoded["word_ids"], encoded2["word_ids"])
        self.assertEqual(encoded["char_ids"], encoded2["char_ids"])
        self.assertNotEqual(encoded["slot_ids"], encoded2["slot_ids"])
        with self.assertRaises(ValueError):
            build_vocab(self.train + [dev], verify_labels=False)

    def test_counts_and_batch_padding_preserve_all_tokens(self):
        self.assertEqual(self.vocab["word_counts"][self.vocab["word_to_id"]["alex"]], 2)
        dataset = MassiveDataset(self.train, self.vocab)
        batch = collate_batch([dataset[0], dataset[1]])
        self.assertEqual(batch["lengths"].tolist(), [2, 3])
        self.assertEqual(batch["word_ids"].shape, (2, 3))
        self.assertEqual(batch["word_ids"][0, 2].item(), 0)
        self.assertEqual(batch["slot_ids"][0, 2].item(), -100)
        self.assertFalse(batch["mask"][0, 2].item())
        self.assertEqual(batch["gold_pairs"][0], {("person", "alex")})
        self.assertEqual(batch["utts"], [row["utt"] for row in self.train])

    def test_official_partition_membership_is_preserved(self):
        rows = self.train + [record("2", "tell alex", "tell [person : alex]", partition="test")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            splits = load_splits(path, verify_counts=False)
            self.assertEqual([row["id"] for row in splits["train"]], ["0", "1"])
            self.assertEqual([row["id"] for row in splits["test"]], ["2"])
            self.assertEqual(splits["dev"], [])
            with self.assertRaises(ValueError):
                load_splits(path)

    def test_embeddings_use_training_rows_only_and_seed_unknowns(self):
        rows = [record("0", "hello u.s. rare", "hello u.s. rare")]
        vocab = build_vocab(rows, verify_labels=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glove.2d.txt"
            path.write_text("hello 1.0 2.0\nus 3.0 4.0\nexternal 8.0 9.0\n", encoding="utf-8")
            matrix, coverage = load_embedding_matrix(path, vocab, 2, seed=13)
            again, _ = load_embedding_matrix(path, vocab, 2, seed=13)
        self.assertEqual(matrix.shape, (len(vocab["words"]), 2))
        np.testing.assert_array_equal(matrix, again)
        np.testing.assert_array_equal(matrix[0], np.zeros(2))
        np.testing.assert_array_equal(matrix[vocab["word_to_id"]["u.s."]], [3.0, 4.0])
        self.assertEqual(coverage["matched_types"], 2)
        self.assertEqual(coverage["external_vocabulary_rows_added"], 0)
        self.assertNotIn("external", vocab["word_to_id"])


if __name__ == "__main__":
    unittest.main()
