"""Small joint intent and BIO slot tagger for the AI-assisted stage.

Inputs are integer IDs for BPE subtokens. The slot head emits logits at each
subtoken; train and read the logits at each word's first subtoken to obtain one
BIO tag per original word for ``sc4002.eval.SpanDecoder``. Initialize randomly
and fit on the official English training split only.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ModelOutput(NamedTuple):
    intent_logits: Tensor  # [batch, num_intents]
    slot_logits: Tensor  # [batch, subtokens, num_slot_tags]
    loss: Tensor | None = None


class ScaleUpJointTransformer(nn.Module):
    """Shared Transformer encoder with separate intent and token heads.

    PyTorch's ``TransformerEncoderLayer`` uses optimized scaled dot-product
    attention where supported. Set ``pad_id`` to the vocabulary's padding ID;
    padded positions are excluded from attention, intent pooling, and slot loss.
    """

    def __init__(
        self,
        vocab_size: int,
        num_intents: int = 60,
        num_slot_tags: int = 111,  # O plus B/I for each of 55 slot types
        *,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 5,
        dim_feedforward: int = 1792,
        max_words: int = 128,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_parameters: int = 20_000_000,
    ) -> None:
        super().__init__()
        if min(
            vocab_size, num_intents, num_slot_tags, d_model, nhead,
            num_layers, dim_feedforward, max_words,
        ) < 1:
            raise ValueError("Model sizes must be positive")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0 <= pad_id < vocab_size:
            raise ValueError("pad_id must be a valid vocabulary ID")

        self.pad_id = pad_id
        self.max_words = max_words
        self.word_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_words, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.intent_head = nn.Linear(d_model, num_intents)
        self.slot_head = nn.Linear(d_model, num_slot_tags)

        if self.parameter_count() > max_parameters:
            raise ValueError(
                f"Model has {self.parameter_count():,} parameters; "
                f"limit is {max_parameters:,}"
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        intent_labels: Tensor | None = None,
        slot_labels: Tensor | None = None,
    ) -> ModelOutput:
        """Return logits and, when labels are supplied, the sum of task losses.

        ``input_ids`` and ``slot_labels`` have shape ``[batch, subtokens]``.
        ``attention_mask`` is true for real words. Use -100 for ignored slot
        labels; padding is ignored automatically, even if labelled otherwise.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, words]")
        batch_size, words = input_ids.shape
        if batch_size == 0 or words == 0 or words > self.max_words:
            raise ValueError(f"Batch must contain 1..{self.max_words} words per example")

        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_id)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            attention_mask = attention_mask.bool()
        if not attention_mask.any(dim=1).all():
            raise ValueError("Every example needs at least one unmasked word")

        positions = torch.arange(words, device=input_ids.device)
        hidden = self.embedding_dropout(
            self.word_embedding(input_ids) + self.position_embedding(positions)
        )
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)

        slot_logits = self.slot_head(hidden)
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True)
        intent_logits = self.intent_head(pooled)

        losses = []
        if intent_labels is not None:
            if intent_labels.shape != (batch_size,):
                raise ValueError("intent_labels must have shape [batch]")
            losses.append(F.cross_entropy(intent_logits, intent_labels))
        if slot_labels is not None:
            if slot_labels.shape != input_ids.shape:
                raise ValueError("slot_labels must match input_ids")
            labels = slot_labels.masked_fill(~attention_mask, -100)
            per_word_loss = F.cross_entropy(
                slot_logits.reshape(-1, slot_logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            losses.append(per_word_loss / labels.ne(-100).sum().clamp_min(1))

        return ModelOutput(intent_logits, slot_logits, sum(losses) if losses else None)
