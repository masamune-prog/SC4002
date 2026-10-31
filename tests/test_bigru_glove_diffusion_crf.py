import torch

from sc4002.models.joint_bigru_glove_diffusion_crf import JointBiGRUGloveDiffusionCRF


def test_diffusion_crf_trains_both_slot_objectives():
    torch.manual_seed(5)
    model = JointBiGRUGloveDiffusionCRF(
        vocab_size=6, glove_dim=2, slot_tags=["O", "B-x", "I-x"],
        num_intents=2, embedding_dim=8, hidden_size=8,
        gru_layers=1, denoiser_layers=1, denoiser_heads=2,
        denoiser_ff=16, diffusion_steps=3,
    )
    ids = torch.tensor([[2, 3], [4, 0]])
    vectors = torch.randn(2, 2, 2)
    mask = torch.tensor([[True, True], [True, False]])
    slots = torch.tensor([[1, 2], [0, 0]])
    loss = model.training_loss(ids, vectors, mask, torch.tensor([0, 1]), slots)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.bio_decoder.transitions.grad is not None
    assert model.slot_head.weight.grad is not None
    model.eval()
    with torch.inference_mode():
        _, paths = model.predict(ids, vectors, mask)
    assert [len(path) for path in paths] == [2, 1]
    assert all(path[0] != 2 for path in paths)
