import torch

from sc4002.models.joint_bigru_glove_crf import JointBiGRUGloveCRF


def test_bigru_ignores_padded_words_and_trains_crf():
    torch.manual_seed(7)
    model = JointBiGRUGloveCRF(
        vocab_size=5, glove_dim=2, slot_tags=["O", "B-x", "I-x"],
        num_intents=2, embedding_dim=8, hidden_size=8, num_layers=2,
    )
    model.eval()
    ids = torch.tensor([[2, 3, 0], [4, 0, 0]])
    vectors = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    first = model(ids, vectors, mask)
    changed = vectors.clone()
    changed[~mask] = 1000
    second = model(ids, changed, mask)
    assert torch.allclose(first.intent_logits, second.intent_logits)
    assert torch.allclose(first.slot_emissions[mask], second.slot_emissions[mask])
    slots = torch.tensor([[1, 2, 0], [0, 0, 0]])
    loss = model.loss(first, torch.tensor([0, 1]), slots, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert [len(path) for path in model.decode(first, mask)] == [2, 1]
