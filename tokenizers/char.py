"""
Посимвольный токенизатор (базовый вариант из gpttest / microgpt_with_sberquad).
Экспортирует единый контракт tokenize(data: Dataset) -> Tokenized.
"""

from structs import Dataset, Tokenized


def tokenize(data: Dataset) -> Tokenized:
    # место под токенизацию (построение словаря, BOS и кодирование)
    vocab = sorted(set("".join(data.docs)))
    bos = len(vocab)
    index = {char: pos for pos, char in enumerate(vocab)}
    sequences = [
        [bos] + [index[ch] for ch in doc if ch in index] + [bos]
        for doc in data.docs
    ]
    return Tokenized(
        vocab=vocab,
        bos=bos,
        sequences=sequences,
        decode_token=lambda token_id: (
            vocab[token_id] if 0 <= token_id < len(vocab) else ""
        ),
    )
