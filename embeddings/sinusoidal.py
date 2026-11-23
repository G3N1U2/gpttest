"""
Эмбеддинг wte[token_id] + фиксированное синусоидальное позиционное кодирование.
Не требует обучаемой матрицы wpe (чистый Python, только math).
"""

import math
from autograd import init_matrix
from structs import Embedding, Sizes, Tokenized


def embed(tokens: Tokenized, sizes: Sizes = Sizes()) -> Embedding:
    wte = init_matrix(tokens.vocab_size, sizes.n_embd)
    pe: list[list[float]] = []
    for pos in range(sizes.block_size):
        row: list[float] = []
        for dim in range(sizes.n_embd):
            angle = pos / (10000.0 ** (2 * (dim // 2) / max(1, sizes.n_embd)))
            row.append(0.1 * (math.sin(angle) if dim % 2 == 0 else math.cos(angle)))
        pe.append(row)

    def lookup(token_id: int, pos_id: int):
        token_vec = wte[token_id]
        pos_vec = pe[pos_id % sizes.block_size]
        return [t + p for t, p in zip(token_vec, pos_vec)]

    return Embedding(
        state={"wte": wte},
        lookup=lookup,
        name="sinusoidal",
    )
