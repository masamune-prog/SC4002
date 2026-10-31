"""Word IDs and GloVe lookup derived exclusively from utterance text."""

from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset


def build_word_vocab(examples) -> list[str]:
    """Fit the learned-embedding vocabulary on train utterances only."""
    return ["<pad>", "<unk>"] + sorted({word for ex in examples for word in ex.words})


class GloveLookup:
    def __init__(self, path: Path, words: Iterable[str], dimension: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"GloVe file not found: {path}")
        if dimension < 1:
            raise ValueError("GloVe dimension must be positive")
        wanted = {word.lower() for word in words}
        self.dimension = dimension
        self.vectors: dict[str, torch.Tensor] = {}
        self.zero = torch.zeros(dimension)
        if path.suffix == ".zip":
            archive = ZipFile(path)
            matches = [name for name in archive.namelist()
                       if name.endswith(f".{dimension}d.txt")]
            if len(matches) != 1:
                archive.close()
                raise ValueError(f"Expected one {dimension}d text file in {path}")
            stream = archive.open(matches[0])
        else:
            archive = None
            stream = path.open("rb")
        try:
            for line in stream:
                key, _, values = line.partition(b" ")
                try:
                    word = key.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                if word not in wanted:
                    continue
                vector = torch.tensor([float(value) for value in values.split()])
                if vector.numel() != dimension:
                    raise ValueError(f"GloVe vector for {word!r} is not {dimension}d")
                self.vectors[word] = vector
        finally:
            stream.close()
            if archive is not None:
                archive.close()

    def get(self, word: str) -> torch.Tensor:
        return self.vectors.get(word.lower(), self.zero)


class GloveWordDataset(Dataset):
    def __init__(self, examples, vocab: list[str], lookup: GloveLookup,
                 intents: list[str], tags: list[str]) -> None:
        word_ids = {word: index for index, word in enumerate(vocab)}
        intent_ids = {label: index for index, label in enumerate(intents)}
        tag_ids = {label: index for index, label in enumerate(tags)}
        self.rows = [(
            [word_ids.get(word, 1) for word in ex.words],
            torch.stack([lookup.get(word) for word in ex.words]),
            intent_ids[ex.intent],
            [tag_ids[tag] for tag in ex.bio_tags],
        ) for ex in examples]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate_glove(rows):
    batch = len(rows)
    length = max(len(row[0]) for row in rows)
    dimension = rows[0][1].size(-1)
    ids = torch.zeros((batch, length), dtype=torch.long)
    vectors = torch.zeros((batch, length, dimension))
    intents = torch.empty(batch, dtype=torch.long)
    slots = torch.zeros((batch, length), dtype=torch.long)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    for i, (word_ids, glove, intent, tags) in enumerate(rows):
        count = len(word_ids)
        ids[i, :count] = torch.tensor(word_ids)
        vectors[i, :count] = glove
        intents[i] = intent
        slots[i, :count] = torch.tensor(tags)
        mask[i, :count] = True
    return ids, vectors, intents, slots, mask
