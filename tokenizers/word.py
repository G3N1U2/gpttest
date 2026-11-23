"""
Словесный токенизатор на чистом Python (стандартный модуль re).
Экспортирует единый контракт tokenize(data: Dataset) -> Tokenized.
"""

import re
from structs import Dataset, Tokenized

TOKEN_PATTERN = re.compile(r"[\wа-яА-ЯёЁ]+|[^\w\s]")


def tokenize(data: Dataset) -> Tokenized:
    words: list[str] = []
    for doc in data.docs:
        words.extend(TOKEN_PATTERN.findall(doc.lower()))

    vocab = list(dict.fromkeys(words))
    bos = len(vocab)
    index = {word: pos for pos, word in enumerate(vocab)}

    sequences = [
        [bos]
        + [index[w] for w in TOKEN_PATTERN.findall(doc.lower()) if w in index]
        + [bos]
        for doc in data.docs
    ]

    return Tokenized(
        vocab=vocab,
        bos=bos,
        sequences=sequences,
        decode_token=lambda token_id: (
            vocab[token_id] + " " if 0 <= token_id < len(vocab) else ""
        ),
    )
