from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import torch

from sc4002.models.joint_transformer_glove_crf import JointTransformerGloveCRF
from sc4002.tokenizers.glove_words import (
    GloveLookup, GloveWordDataset, build_word_vocab, collate_glove,
)


def test_glove_lookup_keeps_unseen_word_features(tmp_path: Path):
    archive_path = tmp_path / "glove.6B.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("glove.6B.2d.txt", "hello 1 2\nunseen 3 4\n")
    train = [SimpleNamespace(words=("hello",), intent="a", bio_tags=("O",))]
    dev = [SimpleNamespace(words=("unseen",), intent="a", bio_tags=("B-x",))]
    vocab = build_word_vocab(train)
    lookup = GloveLookup(archive_path, ("hello", "unseen"), 2)
    ids, vectors, _, slots, mask = collate_glove(
        [GloveWordDataset(dev, vocab, lookup, ["a"], ["O", "B-x", "I-x"])[0]]
    )
    assert ids.tolist() == [[1]]  # OOV for the learned vocabulary.
    assert vectors.tolist() == [[[3.0, 4.0]]]  # Still has pretrained features.

    model = JointTransformerGloveCRF(
        len(vocab), 2, ["O", "B-x", "I-x"], num_intents=1,
        d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
    )
    output = model(ids, vectors, mask)
    loss = model.loss(output, torch.tensor([0]), slots, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert len(model.decode(output, mask)[0]) == 1
