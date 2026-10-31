"""Word-level Transformer and CRF with external, fixed GloVe features."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sc4002.models.joint_transformer_crf import CRFOutput, LinearChainCRF


class JointTransformerGloveCRF(nn.Module):
    """Combine GloVe features with learned word embeddings before encoding."""

    def __init__(
        self, vocab_size: int, glove_dim: int, slot_tags: Sequence[str],
        num_intents: int = 60, *, d_model: int = 256, nhead: int = 4,
        num_layers: int = 3, dim_feedforward: int = 512,
        max_words: int = 128, dropout: float = 0.1, pad_id: int = 0,
        max_parameters: int = 20_000_000,
    ) -> None:
        super().__init__()
        if min(vocab_size, glove_dim, num_intents, d_model, nhead,
               num_layers, dim_feedforward, max_words) < 1:
            raise ValueError("Model dimensions must be positive")
        if d_model % nhead or not 0 <= pad_id < vocab_size:
            raise ValueError("Invalid attention heads or padding ID")
        self.pad_id = pad_id
        self.max_words = max_words
        self.glove_dim = glove_dim
        self.glove_projection = nn.Linear(glove_dim, d_model, bias=False)
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.zeros_(self.word_embedding.weight)
        self.position_embedding = nn.Embedding(max_words, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.intent_head = nn.Linear(d_model, num_intents)
        self.slot_head = nn.Linear(d_model, len(slot_tags))
        self.crf = LinearChainCRF(slot_tags)
        if self.parameter_count() > max_parameters:
            raise ValueError(f"Model exceeds {max_parameters:,} parameters")

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, word_ids: Tensor, glove_vectors: Tensor, mask: Tensor) -> CRFOutput:
        if word_ids.ndim != 2 or mask.shape != word_ids.shape:
            raise ValueError("word_ids and mask must be [batch, words]")
        if glove_vectors.shape != (*word_ids.shape, self.glove_dim):
            raise ValueError("GloVe vectors must match word IDs and glove_dim")
        if word_ids.size(1) > self.max_words or not mask[:, 0].all():
            raise ValueError("Invalid word sequence length or empty utterance")
        positions = torch.arange(word_ids.size(1), device=word_ids.device)
        hidden = self.embedding_dropout(
            self.glove_projection(glove_vectors) + self.word_embedding(word_ids)
            + self.position_embedding(positions)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return CRFOutput(self.intent_head(pooled), self.slot_head(hidden))

    def loss(
        self, output: CRFOutput, intent_labels: Tensor, slot_labels: Tensor,
        mask: Tensor, *, slot_weight: float = 1.0,
    ) -> Tensor:
        if slot_weight <= 0:
            raise ValueError("slot_weight must be positive")
        return F.cross_entropy(output.intent_logits, intent_labels) + slot_weight * (
            self.crf.neg_log_likelihood(output.slot_emissions, slot_labels, mask)
        )

    def decode(self, output: CRFOutput, mask: Tensor) -> list[list[int]]:
        return self.crf.decode(output.slot_emissions, mask)
