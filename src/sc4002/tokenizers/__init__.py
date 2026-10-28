"""Tokenizers trained solely on the assignment's training split."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sc4002.tokenizers.bpe_tokenizer import BpeTokenizer

__all__ = ["BpeTokenizer"]


def __getattr__(name: str):
    if name == "BpeTokenizer":
        from sc4002.tokenizers.bpe_tokenizer import BpeTokenizer

        return BpeTokenizer
    raise AttributeError(name)
