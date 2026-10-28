"""Train a BPE tokenizer from MASSIVE's English training utterances only."""

import argparse
import csv
from collections.abc import Iterator, Sequence
from pathlib import Path

from tokenizers import Encoding, Tokenizer, decoders, models, pre_tokenizers, trainers

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
WORD_SUFFIX = "</w>"
CONTINUATION_PREFIX = "##"


def _training_utterances(csv_path: Path) -> Iterator[str]:
    with csv_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"utt", "partition", "locale"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain {sorted(required)} columns")
        for row_number, row in enumerate(reader, start=2):
            if row["partition"] != "train" or row["locale"] != "en-US":
                raise ValueError(f"Row {row_number} is not in the English training split")
            utterance = row["utt"]
            if not utterance or not utterance.strip():
                raise ValueError(f"Row {row_number} has an empty utt")
            yield utterance


class BpeTokenizer:
    """Whitespace-aligned byte-level BPE for the word-level slot task.

    ``encode`` returns a Tokenizers ``Encoding``. Its ``word_ids`` map each
    subword token back to the index in ``utterance.split()``, so a model can
    align subword features with one BIO label per original word.
    """

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        if self.pad_id is None or self.unk_id is None:
            raise ValueError("Tokenizer must contain [PAD] and [UNK]")

    @classmethod
    def train_from_csv(
        cls,
        csv_path: str | Path = "data/train.csv",
        *,
        vocab_size: int = 10_000,
        min_frequency: int = 1,
        expected_rows: int | None = 11_514,
    ) -> "BpeTokenizer":
        """Fit only on ``utt``; validate the official split and row count."""
        if vocab_size < 2 or min_frequency < 1:
            raise ValueError("vocab_size must be >= 2 and min_frequency >= 1")
        csv_path = Path(csv_path)
        rows = sum(1 for _ in _training_utterances(csv_path))
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"Expected {expected_rows} training rows, got {rows}")

        tokenizer = Tokenizer(
            models.BPE(
                unk_token=UNK_TOKEN,
                continuing_subword_prefix=CONTINUATION_PREFIX,
                end_of_word_suffix=WORD_SUFFIX,
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.Sequence(
            [
                decoders.Replace(CONTINUATION_PREFIX, ""),
                decoders.BPEDecoder(suffix=WORD_SUFFIX),
                decoders.ByteLevel(),
            ]
        )
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=[PAD_TOKEN, UNK_TOKEN],
            continuing_subword_prefix=CONTINUATION_PREFIX,
            end_of_word_suffix=WORD_SUFFIX,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(
            _training_utterances(csv_path), trainer=trainer, length=rows
        )
        return cls(tokenizer)

    @property
    def pad_id(self) -> int | None:
        return self.tokenizer.token_to_id(PAD_TOKEN)

    @property
    def unk_id(self) -> int | None:
        return self.tokenizer.token_to_id(UNK_TOKEN)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, utterance: str) -> Encoding:
        return self.tokenizer.encode(utterance, add_special_tokens=False)

    @staticmethod
    def align_word_labels(
        encoding: Encoding, word_labels: Sequence[int], ignore_index: int = -100
    ) -> list[int]:
        """Put each word's BIO label on its first subtoken for model training."""
        word_ids = encoding.word_ids
        word_count = max((i for i in word_ids if i is not None), default=-1) + 1
        if len(word_labels) != word_count:
            raise ValueError(f"Expected {word_count} word labels, got {len(word_labels)}")
        aligned = []
        previous = None
        for word_id in word_ids:
            if word_id is None or word_id == previous:
                aligned.append(ignore_index)
            else:
                aligned.append(word_labels[word_id])
            previous = word_id
        return aligned

    @staticmethod
    def collapse_subword_labels(
        encoding: Encoding, subword_labels: Sequence[int]
    ) -> list[int]:
        """Take one prediction per original word for the span decoder."""
        word_ids = encoding.word_ids
        if len(subword_labels) != len(word_ids):
            raise ValueError(f"Expected {len(word_ids)} subword labels")
        word_labels = []
        previous = None
        for word_id, label in zip(word_ids, subword_labels):
            if word_id is not None and word_id != previous:
                word_labels.append(label)
            previous = word_id
        return word_labels

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BpeTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def to_json(self) -> str:
        """Serialize the exact fitted tokenizer for inclusion in a checkpoint."""
        return self.tokenizer.to_str()

    @classmethod
    def from_json(cls, payload: str) -> "BpeTokenizer":
        return cls(Tokenizer.from_str(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE on MASSIVE en-US train.csv")
    parser.add_argument("--csv", default="data/train.csv")
    parser.add_argument("--output", default="data/bpe_tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--min-frequency", type=int, default=1)
    args = parser.parse_args()
    tokenizer = BpeTokenizer.train_from_csv(
        args.csv, vocab_size=args.vocab_size, min_frequency=args.min_frequency
    )
    tokenizer.save(args.output)
    print(f"Saved {tokenizer.vocab_size} tokens to {args.output}")


if __name__ == "__main__":
    main()
