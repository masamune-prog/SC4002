"""Word-level GloVe + bidirectional GRU with intent and BIO CRF heads."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sc4002.models.joint_transformer_crf import CRFOutput, LinearChainCRF


class JointBiGRUGloveCRF(nn.Module):
    """Use cuDNN GRU when available and mask padding before both task heads."""

    def __init__(
        self, vocab_size: int, glove_dim: int, slot_tags: Sequence[str],
        num_intents: int = 60, *, embedding_dim: int = 256,
        hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.1,
        max_words: int = 128, pad_id: int = 0,
        max_parameters: int = 20_000_000,
    ) -> None:
        super().__init__()
        if min(vocab_size, glove_dim, num_intents, embedding_dim,
               hidden_size, num_layers, max_words) < 1:
            raise ValueError("Model dimensions must be positive")
        if not 0 <= pad_id < vocab_size or not 0 <= dropout < 1:
            raise ValueError("Invalid padding ID or dropout")
        self.glove_dim = glove_dim
        self.max_words = max_words
        self.glove_projection = nn.Linear(glove_dim, embedding_dim, bias=False)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        nn.init.zeros_(self.word_embedding.weight)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(2 * hidden_size, num_intents)
        self.slot_head = nn.Linear(2 * hidden_size, len(slot_tags))
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
        if (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("mask must mark a contiguous word prefix")
        embeddings = self.embedding_dropout(
            self.glove_projection(glove_vectors) + self.word_embedding(word_ids)
        )
        lengths = mask.sum(dim=1).cpu()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True,
                                      enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True,
                                        total_length=word_ids.size(1))
        hidden = self.output_dropout(hidden)
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
