"""Joint intent classifier and word-level BIO tagger with a linear-chain CRF."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sc4002.models.joint_transformer import JointTransformer


class CRFOutput(NamedTuple):
    intent_logits: Tensor  # [batch, intents]
    slot_emissions: Tensor  # [batch, words, BIO tags]


class LinearChainCRF(nn.Module):
    """Batched CRF whose start and transition masks enforce valid BIO paths."""

    def __init__(self, slot_tags: Sequence[str]) -> None:
        super().__init__()
        if not slot_tags or slot_tags[0] != "O" or len(set(slot_tags)) != len(slot_tags):
            raise ValueError("slot_tags must start with O and contain unique tags")
        for tag in slot_tags[1:]:
            if not (tag.startswith("B-") or tag.startswith("I-")) or len(tag) < 3:
                raise ValueError(f"Invalid BIO tag: {tag!r}")

        count = len(slot_tags)
        self.transitions = nn.Parameter(torch.empty(count, count))
        self.start_transitions = nn.Parameter(torch.empty(count))
        self.end_transitions = nn.Parameter(torch.empty(count))
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

        start_allowed = torch.tensor([not tag.startswith("I-") for tag in slot_tags])
        transition_allowed = torch.ones(count, count, dtype=torch.bool)
        for previous, previous_tag in enumerate(slot_tags):
            for current, current_tag in enumerate(slot_tags):
                if current_tag.startswith("I-"):
                    slot_type = current_tag[2:]
                    transition_allowed[previous, current] = previous_tag in {
                        f"B-{slot_type}", f"I-{slot_type}"
                    }
        self.register_buffer("start_allowed", start_allowed)
        self.register_buffer("transition_allowed", transition_allowed)

    def _scores(self) -> tuple[Tensor, Tensor]:
        start = self.start_transitions.masked_fill(~self.start_allowed, -1e4)
        transitions = self.transitions.masked_fill(~self.transition_allowed, -1e4)
        return start, transitions

    @staticmethod
    def _check(emissions: Tensor, mask: Tensor) -> None:
        if emissions.ndim != 3 or mask.shape != emissions.shape[:2]:
            raise ValueError("emissions must be [batch, words] and mask [batch, words]")
        if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("mask must mark a nonempty contiguous word prefix")

    def neg_log_likelihood(self, emissions: Tensor, tags: Tensor, mask: Tensor) -> Tensor:
        """Mean negative log likelihood across utterances."""
        mask = mask.bool()
        self._check(emissions, mask)
        if tags.shape != mask.shape:
            raise ValueError("tags and mask must have the same shape")
        start, transitions = self._scores()
        safe_tags = tags.masked_fill(~mask, 0)
        if (safe_tags < 0).any() or (safe_tags >= emissions.size(-1)).any():
            raise ValueError("tags contain an invalid ID")

        alpha = start + emissions[:, 0]
        for step in range(1, emissions.size(1)):
            candidates = alpha.unsqueeze(2) + transitions.unsqueeze(0)
            next_alpha = torch.logsumexp(candidates, dim=1) + emissions[:, step]
            alpha = torch.where(mask[:, step, None], next_alpha, alpha)
        log_partition = torch.logsumexp(alpha + self.end_transitions, dim=1)

        first_tags = safe_tags[:, 0]
        path_score = start[first_tags] + emissions[:, 0].gather(1, first_tags[:, None]).squeeze(1)
        for step in range(1, emissions.size(1)):
            previous, current = safe_tags[:, step - 1], safe_tags[:, step]
            emission = emissions[:, step].gather(1, current[:, None]).squeeze(1)
            path_score = path_score + (transitions[previous, current] + emission) * mask[:, step]
        last_positions = mask.long().sum(dim=1) - 1
        last_tags = safe_tags.gather(1, last_positions[:, None]).squeeze(1)
        path_score = path_score + self.end_transitions[last_tags]
        return (log_partition - path_score).mean()

    @torch.no_grad()
    def decode(self, emissions: Tensor, mask: Tensor) -> list[list[int]]:
        """Viterbi decode one valid BIO path per utterance."""
        mask = mask.bool()
        self._check(emissions, mask)
        start, transitions = self._scores()
        scores = start + emissions[:, 0]
        backpointers = []
        for step in range(1, emissions.size(1)):
            candidates = scores.unsqueeze(2) + transitions.unsqueeze(0)
            best_scores, best_previous = candidates.max(dim=1)
            next_scores = best_scores + emissions[:, step]
            scores = torch.where(mask[:, step, None], next_scores, scores)
            backpointers.append(best_previous)
        final_tags = (scores + self.end_transitions).argmax(dim=1).tolist()
        lengths = mask.long().sum(dim=1).tolist()
        paths = []
        for row, length in enumerate(lengths):
            path = [final_tags[row]]
            for step in range(length - 2, -1, -1):
                path.append(int(backpointers[step][row, path[-1]]))
            paths.append(path[::-1])
        return paths


class JointTransformerCRF(nn.Module):
    """Shared Transformer with an intent head and word-level BIO CRF head."""

    def __init__(
        self, vocab_size: int, slot_tags: Sequence[str], num_intents: int = 60,
        *, d_model: int = 384, nhead: int = 6, num_layers: int = 9,
        dim_feedforward: int = 1536, norm_first: bool = True,
        max_parameters: int = 20_000_000, **encoder_kwargs,
    ) -> None:
        super().__init__()
        self.backbone = JointTransformer(
            vocab_size=vocab_size,
            num_intents=num_intents,
            num_slot_tags=len(slot_tags),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            norm_first=norm_first,
            max_parameters=max_parameters,
            **encoder_kwargs,
        )
        self.crf = LinearChainCRF(slot_tags)
        if self.parameter_count() > max_parameters:
            raise ValueError(
                f"Model has {self.parameter_count():,} parameters; limit is {max_parameters:,}"
            )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, input_ids: Tensor, word_positions: Tensor, word_mask: Tensor
    ) -> CRFOutput:
        """Use token IDs and word starts derived exclusively from ``utt``."""
        if word_positions.ndim != 2 or word_positions.shape != word_mask.shape:
            raise ValueError("word_positions and word_mask must be [batch, words]")
        if word_positions.size(0) != input_ids.size(0):
            raise ValueError("word_positions batch size must match input_ids")
        if (word_positions < 0).any() or (word_positions >= input_ids.size(1)).any():
            raise ValueError("word_positions contain an invalid subtoken index")
        output = self.backbone(input_ids)
        index = word_positions.unsqueeze(-1).expand(-1, -1, output.slot_logits.size(-1))
        emissions = output.slot_logits.gather(1, index)
        return CRFOutput(output.intent_logits, emissions)

    def loss(
        self, output: CRFOutput, intent_labels: Tensor, slot_labels: Tensor,
        word_mask: Tensor, *, slot_weight: float = 1.0,
    ) -> Tensor:
        if slot_weight <= 0:
            raise ValueError("slot_weight must be positive")
        intent_loss = F.cross_entropy(output.intent_logits, intent_labels)
        slot_loss = self.crf.neg_log_likelihood(output.slot_emissions, slot_labels, word_mask)
        return intent_loss + slot_weight * slot_loss

    def decode(self, output: CRFOutput, word_mask: Tensor) -> list[list[int]]:
        return self.crf.decode(output.slot_emissions, word_mask)
