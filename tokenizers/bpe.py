"""
Byte-Pair Encoding (BPE) токенизатор на чистом Python (только stdlib).
Экспортирует единый контракт tokenize(data: Dataset) -> Tokenized.
"""

from collections import Counter
from structs import Dataset, Tokenized


def _merge_symbols(symbols: list[str], a: str, b: str) -> list[str]:
    merged = a + b
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def tokenize(data: Dataset, num_merges: int = 48) -> Tokenized:
    corpus = "\n".join(data.docs)
    symbols = list(corpus)
    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        pairs = Counter(
            (a, b) for a, b in zip(symbols, symbols[1:]) if a != "\n" and b != "\n"
        )
        if not pairs:
            break
        best_pair, freq = pairs.most_common(1)[0]
        if freq < 2:
            break
        merges.append(best_pair)
        symbols = _merge_symbols(symbols, best_pair[0], best_pair[1])

    base_chars = sorted(set("".join(data.docs)))
    merged_tokens = [a + b for a, b in merges]
    vocab = list(dict.fromkeys([*base_chars, *merged_tokens]))
    bos = len(vocab)
    index = {tok: pos for pos, tok in enumerate(vocab)}

    sequences: list[list[int]] = []
    for doc in data.docs:
        doc_symbols = list(doc)
        for a, b in merges:
            doc_symbols = _merge_symbols(doc_symbols, a, b)
        ids = [index[s] for s in doc_symbols if s in index]
        sequences.append([bos] + ids + [bos])

    return Tokenized(
        vocab=vocab,
        bos=bos,
        sequences=sequences,
        decode_token=lambda token_id: (
            vocab[token_id] if 0 <= token_id < len(vocab) else ""
        ),
    )
