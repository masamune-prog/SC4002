"""Slot denoising as an auxiliary objective for a GloVe BiGRU + CRF."""

import torch
from torch import Tensor
from torch.nn import functional as F

from sc4002.models.joint_bigru_glove_diffusion import JointBiGRUGloveDiffusion


class JointBiGRUGloveDiffusionCRF(JointBiGRUGloveDiffusion):
    """Train masked-tag denoising jointly with sequence-level CRF likelihood."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for parameter in self.bio_decoder.parameters():
            parameter.requires_grad_(True)

    def training_loss(self, word_ids: Tensor, glove_vectors: Tensor,
                      mask: Tensor, intent_labels: Tensor,
                      slot_labels: Tensor, *, slot_weight: float = 1.0) -> Tensor:
        if slot_weight <= 0 or slot_labels.shape != word_ids.shape:
            raise ValueError("Invalid slot weight or slot label shape")
        hidden, intent_logits = self._encode(word_ids, glove_vectors, mask)
        batch, words = word_ids.shape
        steps = torch.randint(1, self.diffusion_steps + 1, (batch,),
                              device=word_ids.device)
        corrupted = (torch.rand(batch, words, device=word_ids.device)
                     < steps[:, None] / self.diffusion_steps) & mask
        missing = ~corrupted.any(1)
        if missing.any():
            corrupted[missing, 0] = True
        noisy = slot_labels.masked_fill(corrupted | ~mask, self.mask_tag_id)
        denoised = self._denoise(hidden, noisy, steps, mask)
        diffusion_loss = F.cross_entropy(denoised[corrupted], slot_labels[corrupted])

        all_mask = torch.full_like(slot_labels, self.mask_tag_id)
        final_steps = torch.full((batch,), self.diffusion_steps,
                                 device=word_ids.device, dtype=torch.long)
        emissions = self._denoise(hidden, all_mask, final_steps, mask)
        crf_loss = self.bio_decoder.neg_log_likelihood(emissions, slot_labels, mask)
        intent_loss = F.cross_entropy(intent_logits, intent_labels)
        return intent_loss + slot_weight * (crf_loss + 0.5 * diffusion_loss)

    def validation_loss(self, word_ids: Tensor, glove_vectors: Tensor,
                        mask: Tensor, intent_labels: Tensor,
                        slot_labels: Tensor, *, slot_weight: float = 1.0) -> Tensor:
        hidden, intent_logits = self._encode(word_ids, glove_vectors, mask)
        all_mask = torch.full_like(slot_labels, self.mask_tag_id)
        steps = torch.full((len(word_ids),), self.diffusion_steps,
                           device=word_ids.device, dtype=torch.long)
        emissions = self._denoise(hidden, all_mask, steps, mask)
        return (F.cross_entropy(intent_logits, intent_labels)
                + slot_weight * self.bio_decoder.neg_log_likelihood(
                    emissions, slot_labels, mask))

    @torch.no_grad()
    def predict(self, word_ids: Tensor, glove_vectors: Tensor,
                mask: Tensor) -> tuple[Tensor, list[list[int]]]:
        return self.predict_all_mask(word_ids, glove_vectors, mask)
