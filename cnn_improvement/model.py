"""Joint intent and slot model with a shared, non-causal dilated CNN.

All contextual-model weights are randomly initialized. A caller may supply a
matrix of permitted static word vectors. Frozen vectors remain model parameters
and are included by :meth:`JointDilatedCNN.parameter_counts`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BIOCRF(nn.Module):
    """Linear-chain CRF with hard BIO constraints.

``transitions[i, j]`` scores the transition from tag i to tag j. The mask must
describe non-empty, right-padded sequences. Padded gold labels may be -100.
The exact forward and Viterbi algorithms operate over all allowed BIO paths.
"""

    def __init__(self, id2tag: Sequence[str] | Mapping[int, str]):
        super().__init__()
        labels = [id2tag[i] for i in range(len(id2tag))]
        if not labels or any(
            label != "O" and not label.startswith(("B-", "I-")) for label in labels
        ):
            raise ValueError("CRF labels must be O, B-type, or I-type.")
        self.n_tags = len(labels)
        self.transitions = nn.Parameter(torch.zeros(self.n_tags, self.n_tags))
        self.start = nn.Parameter(torch.zeros(self.n_tags))
        self.end = nn.Parameter(torch.zeros(self.n_tags))
        allowed_start = torch.ones(self.n_tags, dtype=torch.bool)
        allowed_transition = torch.ones(self.n_tags, self.n_tags, dtype=torch.bool)
        for j, label in enumerate(labels):
            if label.startswith("I-"):
                allowed_start[j] = False
                for i, previous in enumerate(labels):
                    allowed_transition[i, j] = (
                        previous.startswith(("B-", "I-"))
                        and previous[2:] == label[2:]
                    )
        self.register_buffer("allowed_start", allowed_start)
        self.register_buffer("allowed_transition", allowed_transition)

    def constrained_scores(self) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.transitions.masked_fill(~self.allowed_transition, -torch.inf),
            self.start.masked_fill(~self.allowed_start, -torch.inf),
            self.end,
        )

    def path_score(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Return the unnormalized score for each gold path, shape [batch]."""
        transitions, start, end = self.constrained_scores()
        # Masked positions must not be used to index a tag table, even if the
        # training collator represents ignored labels as -100.
        tags = tags.masked_fill(~mask, 0)
        token_scores = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        score = start[tags[:, 0]] + token_scores[:, 0]
        if tags.size(1) > 1:
            steps = transitions[tags[:, :-1], tags[:, 1:]] + token_scores[:, 1:]
            # torch.where avoids invalid -inf * 0 at padded transitions.
            score = score + torch.where(mask[:, 1:], steps, 0.0).sum(dim=1)
        last_tags = tags.gather(1, (mask.sum(dim=1) - 1).unsqueeze(1)).squeeze(1)
        return score + end[last_tags]

    def log_partition(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Exact log sum of exponentiated scores of all valid BIO paths."""
        transitions, start, end = self.constrained_scores()
        alpha = start + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + transitions.unsqueeze(0)
            next_alpha = torch.logsumexp(scores, dim=1) + emissions[:, t]
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)
        return torch.logsumexp(alpha + end, dim=1)

    def nll(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Per-utterance negative log likelihood, without reduction."""
        return self.log_partition(emissions, mask) - self.path_score(emissions, tags, mask)

    @torch.no_grad()
    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        """Viterbi decode to [batch, time], filling padded positions with zero."""
        transitions, start, end = self.constrained_scores()
        batch_size, time_steps, _ = emissions.shape
        score = start + emissions[:, 0]
        backpointers = []
        identity = torch.arange(self.n_tags, device=emissions.device).expand(batch_size, -1)
        for t in range(1, time_steps):
            best, backpointer = (score.unsqueeze(2) + transitions.unsqueeze(0)).max(dim=1)
            active = mask[:, t].unsqueeze(1)
            score = torch.where(active, best + emissions[:, t], score)
            # Backtracking through inactive positions must preserve the current
            # tag rather than following a pointer computed from padding.
            backpointers.append(torch.where(active, backpointer, identity))
        tag = (score + end).argmax(dim=1)
        path = [tag]
        for backpointer in reversed(backpointers):
            tag = backpointer.gather(1, tag.unsqueeze(1)).squeeze(1)
            path.append(tag)
        return torch.stack(path[::-1], dim=1).masked_fill(~mask, 0)


class CharacterCNN(nn.Module):
    """Character features for each word, with padding-invariant max pooling."""

    def __init__(self, n_chars: int, dim: int, channels: int, kernel_size: int, pad_id: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Character convolution kernel_size must be odd.")
        self.pad_id = pad_id
        self.embedding = nn.Embedding(n_chars, dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(dim, channels, kernel_size, padding=kernel_size // 2)
        self.output_dim = channels

    def forward(self, char_ids: Tensor) -> Tensor:
        batch_size, time_steps, char_steps = char_ids.shape
        flat = char_ids.reshape(batch_size * time_steps, char_steps)
        mask = flat.ne(self.pad_id).unsqueeze(1)
        features = F.gelu(self.conv(self.embedding(flat).transpose(1, 2)))
        features = features.masked_fill(~mask, -torch.inf).amax(dim=2)
        features = torch.where(mask.any(dim=2), features, 0.0)
        return features.reshape(batch_size, time_steps, self.output_dim)


class ResidualDilatedBlock(nn.Module):
    """One bidirectional temporal convolution with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=dilation * (kernel_size // 2), dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        # LayerNorm has a learned bias, so mask *after* normalization. This
        # prevents padded positions from becoming artificial context near ends.
        normed = self.norm(states).masked_fill(~mask.unsqueeze(-1), 0.0)
        update = self.conv(normed.transpose(1, 2)).transpose(1, 2)
        states = states + self.dropout(F.gelu(update))
        return states.masked_fill(~mask.unsqueeze(-1), 0.0)


class JointDilatedCNN(nn.Module):
    """One shared CNN encoder, an intent classifier, and a slot decoder.

The optional intent-conditioned slot head uses a *predicted hidden utterance
representation*, never the gold intent label. The joint loss propagates both
tasks' gradients into the same word embeddings and convolutional encoder.
"""

    DEFAULTS: dict[str, Any] = {
        "emb_dim": 100,
        "freeze_emb": False,
        "word_pad_id": 0,
        "char_pad_id": 0,
        "use_char": True,
        "char_dim": 32,
        "char_filters": 64,
        "char_kernel_size": 3,
        "channels": 192,
        "kernel_size": 3,
        "dilations": [1, 2, 4, 8],
        "dropout": 0.25,
        "embedding_dropout": 0.15,
        "head_dim": 192,
        "intent_conditioning": True,
        "use_crf": True,
        "label_smoothing": 0.05,
        "slot_weight": 1.0,
        "slot_loss_normalization": "token",
    }

    def __init__(
        self,
        config: dict[str, Any],
        n_words: int,
        n_chars: int,
        n_intents: int,
        n_tags: int,
        id2tag: Sequence[str] | Mapping[int, str],
        embedding_matrix: Any = None,
    ):
        super().__init__()
        self.config = {**self.DEFAULTS, **config}
        cfg = self.config
        if len(id2tag) != n_tags:
            raise ValueError("n_tags and id2tag disagree.")
        if cfg["kernel_size"] % 2 != 1 or cfg["kernel_size"] < 1:
            raise ValueError("Convolution kernel_size must be positive and odd.")
        if not cfg["dilations"] or any(d < 1 for d in cfg["dilations"]):
            raise ValueError("At least one positive dilation is required.")
        if cfg["slot_loss_normalization"] not in {"token", "sequence"}:
            raise ValueError("slot_loss_normalization must be token or sequence.")

        self.word_embedding = nn.Embedding(n_words, cfg["emb_dim"], padding_idx=cfg["word_pad_id"])
        if embedding_matrix is not None:
            vectors = torch.as_tensor(embedding_matrix, dtype=self.word_embedding.weight.dtype)
            if vectors.shape != self.word_embedding.weight.shape:
                raise ValueError("Embedding matrix has the wrong shape.")
            with torch.no_grad():
                self.word_embedding.weight.copy_(vectors)
        with torch.no_grad():
            self.word_embedding.weight[cfg["word_pad_id"]].zero_()
        self.word_embedding.weight.requires_grad_(not cfg["freeze_emb"])

        self.char_cnn = None
        input_dim = cfg["emb_dim"]
        if cfg["use_char"]:
            self.char_cnn = CharacterCNN(
                n_chars, cfg["char_dim"], cfg["char_filters"],
                cfg["char_kernel_size"], cfg["char_pad_id"],
            )
            input_dim += cfg["char_filters"]
        self.embedding_dropout = nn.Dropout(cfg["embedding_dropout"])
        self.input_projection = nn.Linear(input_dim, cfg["channels"])
        self.input_norm = nn.LayerNorm(cfg["channels"])
        self.encoder = nn.ModuleList([
            ResidualDilatedBlock(cfg["channels"], cfg["kernel_size"], d, cfg["dropout"])
            for d in cfg["dilations"]
        ])
        self.output_norm = nn.LayerNorm(cfg["channels"])
        self.head_dropout = nn.Dropout(cfg["dropout"])
        self.intent_hidden = nn.Linear(2 * cfg["channels"], cfg["head_dim"])
        self.intent_out = nn.Linear(cfg["head_dim"], n_intents)
        slot_input_dim = cfg["channels"] + (cfg["head_dim"] if cfg["intent_conditioning"] else 0)
        self.slot_hidden = nn.Linear(slot_input_dim, cfg["head_dim"])
        self.slot_out = nn.Linear(cfg["head_dim"], n_tags)
        self.crf = BIOCRF(id2tag) if cfg["use_crf"] else None

    @property
    def receptive_field(self) -> int:
        return 1 + (self.config["kernel_size"] - 1) * sum(self.config["dilations"])

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def forward(
        self, word_ids: Tensor, char_ids: Tensor | None, lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        lengths = lengths.to(device=word_ids.device)
        mask = torch.arange(word_ids.size(1), device=word_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        features = self.word_embedding(word_ids)
        if self.char_cnn is not None:
            if char_ids is None:
                raise ValueError("char_ids is required when use_char is True.")
            features = torch.cat((features, self.char_cnn(char_ids)), dim=-1)
        states = F.gelu(self.input_norm(self.input_projection(self.embedding_dropout(features))))
        states = states.masked_fill(~mask.unsqueeze(-1), 0.0)
        for block in self.encoder:
            states = block(states, mask)
        states = self.output_norm(states).masked_fill(~mask.unsqueeze(-1), 0.0)

        # Pool the stable encoder states before head dropout. Dropping individual
        # token states before max pooling shifts the pooled distribution.
        mean_state = states.sum(dim=1) / lengths.clamp_min(1).unsqueeze(1)
        max_state = states.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        utterance = F.gelu(self.intent_hidden(self.head_dropout(torch.cat((mean_state, max_state), dim=-1))))
        intent_logits = self.intent_out(self.head_dropout(utterance))
        slot_features = states
        if self.config["intent_conditioning"]:
            slot_features = torch.cat(
                (states, utterance.unsqueeze(1).expand(-1, states.size(1), -1)), dim=-1,
            )
        slot_states = F.gelu(self.slot_hidden(self.head_dropout(slot_features)))
        emissions = self.slot_out(self.head_dropout(slot_states))
        return intent_logits, emissions, mask

    def compute_loss(
        self,
        intent_logits: Tensor,
        emissions: Tensor,
        mask: Tensor,
        intent_labels: Tensor,
        slot_tags: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return total loss and detached component scalars for logging."""
        intent_loss = F.cross_entropy(
            intent_logits, intent_labels,
            label_smoothing=self.config["label_smoothing"],
        )
        if self.crf is not None:
            sequence_losses = self.crf.nll(emissions, slot_tags, mask)
            if self.config["slot_loss_normalization"] == "token":
                slot_loss = sequence_losses.sum() / mask.sum()
            else:
                slot_loss = sequence_losses.mean()
        else:
            tags = slot_tags.masked_fill(~mask, -100)
            token_loss = F.cross_entropy(
                emissions.transpose(1, 2), tags, ignore_index=-100, reduction="none",
            )
            denominator = mask.sum() if self.config["slot_loss_normalization"] == "token" else mask.size(0)
            slot_loss = token_loss.sum() / denominator
        loss = intent_loss + self.config["slot_weight"] * slot_loss
        return loss, {"intent_loss": intent_loss.detach(), "slot_loss": slot_loss.detach()}

    @torch.no_grad()
    def decode(self, emissions: Tensor, mask: Tensor) -> Tensor:
        if self.crf is not None:
            return self.crf.decode(emissions, mask)
        return emissions.argmax(dim=-1).masked_fill(~mask, 0)
