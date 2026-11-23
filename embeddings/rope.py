"""
Эмбеддинг с вращательным позиционным кодированием RoPE (Rotary Position Embedding).
Реализован на чистом Python через скаляры Value и модуль math.
"""

import math
from autograd import init_matrix
from structs import Embedding, Sizes, Tokenized


def embed(tokens: Tokenized, sizes: Sizes = Sizes()) -> Embedding:
    wte = init_matrix(tokens.vocab_size, sizes.n_embd)

    def lookup(token_id: int, pos_id: int):
        vec = wte[token_id]
        rotated = []
        for dim in range(0, sizes.n_embd, 2):
            theta = pos_id / (10000.0 ** (dim / max(1, sizes.n_embd)))
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            if dim + 1 < sizes.n_embd:
                x0, x1 = vec[dim], vec[dim + 1]
                rotated.append(x0 * cos_t - x1 * sin_t)
                rotated.append(x0 * sin_t + x1 * cos_t)
            else:
                rotated.append(vec[dim])
        return rotated

    return Embedding(
        state={"wte": wte},
        lookup=lookup,
        name="rope",
    )
