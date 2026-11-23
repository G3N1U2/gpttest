"""Реестр модулей токенизации (блок ТОКЕН). Новые PR добавляются сюда."""

from typing import Callable
from structs import Dataset, Tokenized
from .char import tokenize as tokenize_char
from .bpe import tokenize as tokenize_bpe
from .word import tokenize as tokenize_word

TokenizerFn = Callable[[Dataset], Tokenized]

TOKENIZER_REGISTRY: dict[str, TokenizerFn] = {
    "char": tokenize_char,
    "bpe": tokenize_bpe,
    "word": tokenize_word,
}


def get_tokenizer(name: str) -> TokenizerFn:
    if name not in TOKENIZER_REGISTRY:
        available = ", ".join(sorted(TOKENIZER_REGISTRY.keys()))
        raise KeyError(f"Токенизатор '{name}' не найден. Доступны: {available}")
    return TOKENIZER_REGISTRY[name]


__all__ = [
    "TOKENIZER_REGISTRY",
    "get_tokenizer",
    "tokenize_char",
    "tokenize_bpe",
    "tokenize_word",
]
