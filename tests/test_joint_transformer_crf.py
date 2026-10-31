import itertools

import torch

from sc4002.models import JointTransformerCRF
from sc4002.models.joint_transformer_crf import LinearChainCRF


def test_crf_likelihood_matches_exhaustive_valid_paths():
    tags = ["O", "B-time", "I-time"]
    crf = LinearChainCRF(tags)
    with torch.no_grad():
        crf.transitions.zero_()
        crf.start_transitions.zero_()
        crf.end_transitions.zero_()
    emissions = torch.tensor([[[0.2, 0.4, 4.0], [0.1, 0.3, 0.8]]])
    mask = torch.tensor([[True, True]])
    gold = torch.tensor([[1, 2]])
    valid = [
        path for path in itertools.product(range(3), repeat=2)
        if path[0] != 2 and (path[1] != 2 or path[0] in {1, 2})
    ]
    scores = torch.stack([emissions[0, 0, a] + emissions[0, 1, b] for a, b in valid])
    expected = torch.logsumexp(scores, dim=0) - emissions[0, 0, 1] - emissions[0, 1, 2]
    torch.testing.assert_close(crf.neg_log_likelihood(emissions, gold, mask), expected)
    decoded = crf.decode(emissions, mask)[0]
    assert tuple(decoded) in valid


def test_crf_rejects_cross_type_i_transition():
    crf = LinearChainCRF(["O", "B-time", "I-time", "B-date", "I-date"])
    with torch.no_grad():
        crf.transitions.zero_()
        crf.start_transitions.zero_()
        crf.end_transitions.zero_()
    emissions = torch.tensor([[[0.0, 10.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 10.0]]])
    path = crf.decode(emissions, torch.tensor([[True, True]]))[0]
    assert path != [1, 4]
    assert path[0] != 2 and path[0] != 4


def test_crf_padding_does_not_change_short_sequence():
    crf = LinearChainCRF(["O", "B-time", "I-time"])
    emissions = torch.tensor([
        [[0.4, 0.2, 0.1], [0.1, 0.5, 0.3], [9.0, 9.0, 9.0]],
        [[0.2, 0.4, 0.1], [0.2, 0.1, 0.8], [0.9, 0.1, 0.1]],
    ])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    tags = torch.tensor([[0, 1, 0], [1, 2, 0]])
    batched_loss = crf.neg_log_likelihood(emissions, tags, mask)
    first_loss = crf.neg_log_likelihood(
        emissions[:1, :2], tags[:1, :2], mask[:1, :2]
    )
    second_loss = crf.neg_log_likelihood(
        emissions[1:], tags[1:], mask[1:]
    )
    torch.testing.assert_close(batched_loss, (first_loss + second_loss) / 2)
    assert crf.decode(emissions, mask)[0] == crf.decode(
        emissions[:1, :2], mask[:1, :2]
    )[0]


def test_joint_transformer_crf_forward_and_backward():
    model = JointTransformerCRF(
        vocab_size=32, slot_tags=["O", "B-time", "I-time"],
        num_intents=4, d_model=32, nhead=4, num_layers=1,
        dim_feedforward=64,
    )
    ids = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
    positions = torch.tensor([[0, 2], [0, 3]])
    mask = torch.tensor([[True, True], [True, True]])
    output = model(ids, positions, mask)
    assert output.intent_logits.shape == (2, 4)
    assert output.slot_emissions.shape == (2, 2, 3)
    loss = model.loss(output, torch.tensor([1, 2]), torch.tensor([[0, 1], [1, 2]]), mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.crf.transitions.grad.abs().sum() > 0
