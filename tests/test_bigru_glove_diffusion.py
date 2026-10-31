import torch

from sc4002.models.joint_bigru_glove_diffusion import JointBiGRUGloveDiffusion


def test_masked_diffusion_trains_and_decodes_legal_bio_paths():
    torch.manual_seed(4)
    model = JointBiGRUGloveDiffusion(
        vocab_size=6, glove_dim=2, slot_tags=["O", "B-x", "I-x"],
        num_intents=2, embedding_dim=8, hidden_size=8,
        gru_layers=1, denoiser_layers=1, denoiser_heads=2,
        denoiser_ff=16, diffusion_steps=3,
    )
    ids = torch.tensor([[2, 3, 4], [5, 0, 0]])
    vectors = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    slots = torch.tensor([[0, 1, 2], [0, 0, 0]])
    loss = model.training_loss(ids, vectors, mask, torch.tensor([0, 1]), slots)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.slot_head.weight.grad is not None
    model.eval()
    with torch.inference_mode():
        intent_logits, paths = model.predict(ids, vectors, mask)
    assert intent_logits.shape == (2, 2)
    assert [len(path) for path in paths] == [3, 1]
    assert all(path[0] != 2 for path in paths)
    assert all(not (path[i] == 2 and path[i - 1] not in (1, 2))
               for path in paths for i in range(1, len(path)))
