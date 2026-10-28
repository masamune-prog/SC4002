import importlib.util
import unittest

if importlib.util.find_spec("torch") is not None:
    import torch

    from sc4002.models import JointTransformer
else:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class JointTransformerTests(unittest.TestCase):
    def make_model(self):
        return JointTransformer(
            vocab_size=32, d_model=32, nhead=4, num_layers=1,
            dim_feedforward=64,
        )

    def test_joint_outputs_loss_and_backward(self):
        model = self.make_model()
        input_ids = torch.tensor([[2, 3, 0], [4, 5, 6]])
        output = model(
            input_ids,
            intent_labels=torch.tensor([1, 2]),
            slot_labels=torch.tensor([[0, 1, -100], [0, 2, 3]]),
        )
        self.assertEqual(output.intent_logits.shape, (2, 60))
        self.assertEqual(output.slot_logits.shape, (2, 3, 111))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertGreater(model.word_embedding.weight.grad[2].abs().sum(), 0)
        self.assertEqual(model.word_embedding.weight.grad[0].abs().sum(), 0)

    def test_padding_does_not_change_intent_logits(self):
        model = self.make_model().eval()
        with torch.no_grad():
            padded = model(torch.tensor([[2, 3, 0, 0]])).intent_logits
            unpadded = model(torch.tensor([[2, 3]])).intent_logits
        torch.testing.assert_close(padded, unpadded, atol=1e-6, rtol=1e-6)

    def test_parameter_cap(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            JointTransformer(vocab_size=100, max_parameters=1)
