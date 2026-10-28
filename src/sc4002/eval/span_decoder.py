"""Convert one BIO tag per whitespace-delimited input word into slot spans."""

from collections.abc import Sequence


class SpanDecoder:
    """Decode BIO tags into ``(slot_type, slot_span)`` pairs.

    ``I-type`` after ``O`` or after another type starts a new span. This makes
    malformed model outputs deterministic without silently joining unlike slots.
    """

    def decode(self, utterance: str, tags: Sequence[str]) -> list[tuple[str, str]]:
        words = utterance.split()
        if len(words) != len(tags):
            raise ValueError(f"Expected {len(words)} word tags, got {len(tags)}")

        spans: list[tuple[str, str]] = []
        current_type: str | None = None
        current_words: list[str] = []

        def finish() -> None:
            nonlocal current_type, current_words
            if current_type is not None:
                spans.append((current_type, " ".join(current_words)))
            current_type, current_words = None, []

        for word, tag in zip(words, tags):
            if tag == "O":
                finish()
                continue
            prefix, separator, slot_type = tag.partition("-")
            if not separator or prefix not in {"B", "I"} or not slot_type:
                raise ValueError(f"Invalid BIO tag: {tag!r}")
            if prefix == "B" or current_type != slot_type:
                finish()
                current_type = slot_type
            current_words.append(word)
        finish()
        return spans

    def decode_batch(
        self, utterances: Sequence[str], tag_sequences: Sequence[Sequence[str]]
    ) -> list[list[tuple[str, str]]]:
        if len(utterances) != len(tag_sequences):
            raise ValueError("utterances and tag_sequences must have the same length")
        return [self.decode(utt, tags) for utt, tags in zip(utterances, tag_sequences)]
