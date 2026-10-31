"""Conditional masked diffusion for word-level BIO labels."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sc4002.models.joint_transformer_crf import LinearChainCRF


class JointBiGRUGloveDiffusion(nn.Module):
    """BiGRU intent encoder with a discrete slot-label diffusion head.

    At step t, each gold BIO tag is replaced by MASK with probability t/T.
    The Transformer predicts the original tags only at corrupted positions.
    Inference begins fully masked and commits high-confidence tags gradually.
    A fixed BIO-constrained Viterbi decoder validates the final sequence.
    """

    def __init__(
        self, vocab_size: int, glove_dim: int, slot_tags: Sequence[str],
        num_intents: int = 60, *, embedding_dim: int = 256,
        hidden_size: int = 256, gru_layers: int = 2,
        denoiser_layers: int = 2, denoiser_heads: int = 4,
        denoiser_ff: int = 512, diffusion_steps: int = 6,
        dropout: float = 0.1, max_words: int = 128, pad_id: int = 0,
        max_parameters: int = 20_000_000,
    ) -> None:
        super().__init__()
        if min(vocab_size, glove_dim, num_intents, embedding_dim, hidden_size,
               gru_layers, denoiser_layers, denoiser_heads, denoiser_ff,
               diffusion_steps, max_words) < 1:
            raise ValueError("Model dimensions must be positive")
        if embedding_dim % denoiser_heads or not 0 <= pad_id < vocab_size:
            raise ValueError("Invalid attention heads or padding ID")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.glove_dim = glove_dim
        self.max_words = max_words
        self.diffusion_steps = diffusion_steps
        self.mask_tag_id = len(slot_tags)
        self.glove_projection = nn.Linear(glove_dim, embedding_dim, bias=False)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        nn.init.zeros_(self.word_embedding.weight)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(
            embedding_dim, hidden_size, gru_layers, batch_first=True,
            bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.intent_head = nn.Linear(2 * hidden_size, num_intents)
        self.context_projection = nn.Linear(2 * hidden_size, embedding_dim)
        self.tag_embedding = nn.Embedding(len(slot_tags) + 1, embedding_dim)
        self.step_embedding = nn.Embedding(diffusion_steps + 1, embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=denoiser_heads,
            dim_feedforward=denoiser_ff, dropout=dropout,
            activation="gelu", batch_first=True,
        )
        self.denoiser = nn.TransformerEncoder(layer, denoiser_layers,
                                              enable_nested_tensor=False)
        self.slot_head = nn.Linear(embedding_dim, len(slot_tags))
        # Only legal BIO paths are allowed; transition scores are fixed at zero.
        self.bio_decoder = LinearChainCRF(slot_tags)
        with torch.no_grad():
            self.bio_decoder.transitions.zero_()
            self.bio_decoder.start_transitions.zero_()
            self.bio_decoder.end_transitions.zero_()
        for parameter in self.bio_decoder.parameters():
            parameter.requires_grad_(False)
        if self.parameter_count() > max_parameters:
            raise ValueError(f"Model exceeds {max_parameters:,} parameters")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _encode(self, word_ids: Tensor, glove_vectors: Tensor, mask: Tensor):
        if word_ids.ndim != 2 or mask.shape != word_ids.shape:
            raise ValueError("word_ids and mask must be [batch, words]")
        if glove_vectors.shape != (*word_ids.shape, self.glove_dim):
            raise ValueError("GloVe vectors must match word IDs and glove_dim")
        if word_ids.size(1) > self.max_words or not mask[:, 0].all():
            raise ValueError("Invalid word sequence length or empty utterance")
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("mask must mark a contiguous word prefix")
        embeddings = self.embedding_dropout(
            self.glove_projection(glove_vectors) + self.word_embedding(word_ids)
        )
        packed = pack_padded_sequence(embeddings, mask.sum(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True,
                                        total_length=word_ids.size(1))
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return hidden, self.intent_head(pooled)

    def _denoise(self, hidden: Tensor, noisy_tags: Tensor, steps: Tensor,
                 mask: Tensor) -> Tensor:
        x = (self.context_projection(hidden) + self.tag_embedding(noisy_tags)
             + self.step_embedding(steps).unsqueeze(1))
        x = self.denoiser(x, src_key_padding_mask=~mask)
        return self.slot_head(x)

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
        # Every row must contribute a slot loss, including short utterances.
        missing = ~corrupted.any(1)
        if missing.any():
            corrupted[missing, 0] = True
        noisy_tags = slot_labels.masked_fill(corrupted | ~mask, self.mask_tag_id)
        logits = self._denoise(hidden, noisy_tags, steps, mask)
        slot_loss = F.cross_entropy(logits[corrupted], slot_labels[corrupted])
        intent_loss = F.cross_entropy(intent_logits, intent_labels)
        return intent_loss + slot_weight * slot_loss

    def validation_loss(self, word_ids: Tensor, glove_vectors: Tensor,
                        mask: Tensor, intent_labels: Tensor,
                        slot_labels: Tensor, *, slot_weight: float = 1.0) -> Tensor:
        """Deterministic fully masked loss for a comparable dev history."""
        hidden, intent_logits = self._encode(word_ids, glove_vectors, mask)
        noisy = torch.full_like(slot_labels, self.mask_tag_id)
        steps = torch.full((len(word_ids),), self.diffusion_steps,
                           device=word_ids.device, dtype=torch.long)
        logits = self._denoise(hidden, noisy, steps, mask)
        return (F.cross_entropy(intent_logits, intent_labels)
                + slot_weight * F.cross_entropy(logits[mask], slot_labels[mask]))

    @torch.no_grad()
    def predict_all_mask(self, word_ids: Tensor, glove_vectors: Tensor,
                         mask: Tensor) -> tuple[Tensor, list[list[int]]]:
        """Single denoising step for a decoding ablation."""
        hidden, intent_logits = self._encode(word_ids, glove_vectors, mask)
        noisy = torch.full_like(word_ids, self.mask_tag_id)
        steps = torch.full((len(word_ids),), self.diffusion_steps,
                           device=word_ids.device, dtype=torch.long)
        logits = self._denoise(hidden, noisy, steps, mask)
        return intent_logits, self.bio_decoder.decode(logits, mask)

    @torch.no_grad()
    def predict(self, word_ids: Tensor, glove_vectors: Tensor,
                mask: Tensor) -> tuple[Tensor, list[list[int]]]:
        hidden, intent_logits = self._encode(word_ids, glove_vectors, mask)
        batch, words = word_ids.shape
        noisy = torch.full((batch, words), self.mask_tag_id,
                           device=word_ids.device, dtype=torch.long)
        lengths = mask.sum(1)
        for step in range(self.diffusion_steps, 0, -1):
            times = torch.full((batch,), step, device=word_ids.device,
                               dtype=torch.long)
            logits = self._denoise(hidden, noisy, times, mask)
            confidence, predicted = logits.softmax(-1).max(-1)
            # Commit a growing fraction of tags, retaining previous choices.
            count = torch.div(lengths * (self.diffusion_steps - step + 1)
                              + self.diffusion_steps - 1, self.diffusion_steps,
                              rounding_mode="floor")
            remaining = (noisy == self.mask_tag_id) & mask
            need = (count - (~remaining & mask).sum(1)).clamp_min(0)
            order = confidence.masked_fill(~remaining, -1).argsort(1, descending=True)
            ranks = torch.empty_like(order)
            ranks.scatter_(1, order,
                           torch.arange(words, device=word_ids.device).expand(batch, -1))
            commit = remaining & (ranks < need[:, None])
            noisy = torch.where(commit, predicted, noisy)
        # Denoise once more conditioned on all committed tags, then enforce BIO.
        logits = self._denoise(hidden, noisy, torch.ones_like(lengths), mask)
        return intent_logits, self.bio_decoder.decode(logits, mask)
