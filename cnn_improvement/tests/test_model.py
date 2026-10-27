"""Numerical and behavioral tests; no downloaded data or weights required.

Run from the project root with:
    python -m unittest discover -s cnn_improvement/tests -p test_model.py -v
"""

import itertools
import unittest

import torch

from cnn_improvement.model import BIOCRF, JointDilatedCNN


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_crf_matches_independent_brute_force_with_padding(self):
        labels = ["O", "B-x", "I-x", "B-y", "I-y"]
        crf = BIOCRF(labels).double()
        with torch.no_grad():
            crf.transitions.copy_(torch.randn(5, 5, dtype=torch.double))
            crf.start.copy_(torch.randn(5, dtype=torch.double))
            crf.end.copy_(torch.randn(5, dtype=torch.double))
        emissions = torch.randn(3, 4, 5, dtype=torch.double, requires_grad=True)
        lengths = torch.tensor([4, 2, 1])
        mask = torch.arange(4).unsqueeze(0) < lengths.unsqueeze(1)
        tags = torch.tensor([[1, 2, 0, 3], [3, 4, -100, -100], [0, -100, -100, -100]])
        expected_log_partition, expected_path, expected_gold = [], [], []
        # Constraints are checked using label strings here, independently of the
        # CRF's buffers and dynamic-programming implementation.
        for b, length in enumerate(lengths.tolist()):
            paths, scores = [], []
            for path in itertools.product(range(5), repeat=length):
                if labels[path[0]].startswith("I-"):
                    continue
                valid = all(
                    not labels[path[t]].startswith("I-")
                    or labels[path[t - 1]] in {"B-" + labels[path[t]][2:], labels[path[t]]}
                    for t in range(1, length)
                )
                if not valid:
                    continue
                score = crf.start[path[0]] + crf.end[path[-1]]
                for t, tag in enumerate(path):
                    score = score + emissions[b, t, tag]
                    if t:
                        score = score + crf.transitions[path[t - 1], tag]
                paths.append(path)
                scores.append(score)
            scores = torch.stack(scores)
            expected_log_partition.append(torch.logsumexp(scores, dim=0))
            expected_path.append(paths[int(scores.argmax())])
            gold = tuple(tags[b, :length].tolist())
            expected_gold.append(scores[paths.index(gold)])
        log_partition = crf.log_partition(emissions, mask)
        torch.testing.assert_close(log_partition, torch.stack(expected_log_partition), rtol=1e-12, atol=1e-12)
        nll = crf.nll(emissions, tags, mask)
        expected_nll = torch.stack(expected_log_partition) - torch.stack(expected_gold)
        torch.testing.assert_close(nll, expected_nll, rtol=1e-12, atol=1e-12)
        predicted = crf.decode(emissions, mask)
        for b, length in enumerate(lengths.tolist()):
            self.assertEqual(tuple(predicted[b, :length].tolist()), expected_path[b])
            self.assertTrue(torch.all(predicted[b, length:] == 0))
        actual_gradient = torch.autograd.grad(nll.sum(), emissions, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected_nll.sum(), emissions)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-10)
        self.assertTrue(torch.isfinite(actual_gradient).all())
        self.assertTrue(torch.all(actual_gradient[~mask] == 0))

    @staticmethod
    def build_model(**updates):
        config = {
            "emb_dim": 12, "channels": 16, "char_dim": 8, "char_filters": 8,
            "head_dim": 16, "dropout": 0.0, "embedding_dropout": 0.0,
            "dilations": [1, 2, 4], "label_smoothing": 0.0,
            **updates,
        }
        return JointDilatedCNN(config, 30, 20, 3, 3, ["O", "B-x", "I-x"])

    def test_encoder_and_character_pooling_are_padding_invariant(self):
        model = self.build_model().eval()
        short_words = torch.tensor([[2, 3, 4]])
        short_chars = torch.tensor([[[2, 3, 0], [4, 5, 6], [7, 0, 0]]])
        short = model(short_words, short_chars, torch.tensor([3]))
        words = torch.tensor([[2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 11]])
        chars = torch.zeros(2, 7, 6, dtype=torch.long)
        chars[0, :3, :3] = short_chars[0]
        chars[1] = torch.tensor([[2, 3, 4, 5, 6, 7]] * 7)
        batched = model(words, chars, torch.tensor([3, 7]))
        torch.testing.assert_close(short[0][0], batched[0][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(short[1][0], batched[1][0, :3], rtol=1e-5, atol=1e-6)
        self.assertEqual(model.receptive_field, 15)

    def test_both_tasks_train_the_same_encoder(self):
        model = self.build_model()
        words = torch.tensor([[2, 3, 4], [5, 6, 0]])
        chars = torch.tensor([[[2, 3], [4, 0], [5, 0]], [[6, 7], [8, 0], [0, 0]]])
        intent_logits, emissions, mask = model(words, chars, torch.tensor([3, 2]))
        tags = torch.tensor([[1, 2, 0], [0, 1, -100]])
        intent_labels = torch.tensor([1, 2])
        shared_weight = model.encoder[0].conv.weight
        intent_loss = torch.nn.functional.cross_entropy(intent_logits, intent_labels)
        slot_loss = model.crf.nll(emissions, tags, mask).mean()
        for loss in [intent_loss, slot_loss]:
            gradient = torch.autograd.grad(loss, shared_weight, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        total, parts = model.compute_loss(intent_logits, emissions, mask, intent_labels, tags)
        torch.testing.assert_close(total.detach(), parts["intent_loss"] + parts["slot_loss"])
        total.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_frozen_embeddings_still_count_and_roundtrip_checkpoint(self):
        model = self.build_model(freeze_emb=True).eval()
        counts = model.parameter_counts()
        self.assertEqual(counts["total"] - counts["trainable"], 30 * 12)
        self.assertLess(counts["total"], 20_000_000)
        self.assertFalse(model.word_embedding.weight.requires_grad)
        clone = self.build_model(freeze_emb=True).eval()
        clone.load_state_dict(model.state_dict())
        words = torch.tensor([[2, 3]])
        chars = torch.tensor([[[2, 0], [3, 4]]])
        inputs = (words, chars, torch.tensor([2]))
        for actual, expected in zip(clone(*inputs), model(*inputs)):
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
