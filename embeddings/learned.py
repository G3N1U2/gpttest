"""
Базовый обучаемый эмбеддинг wte[token_id] + wpe[pos_id] из gpttest.
Экспортирует единый контракт embed(tokens: Tokenized, sizes: Sizes) -> Embedding.
"""

from autograd import init_matrix
from structs import Embedding, Sizes, Tokenized


def embed(tokens: Tokenized, sizes: Sizes = Sizes()) -> Embedding:
    # место под эмбеддинг (инициализация обучаемых матриц wte и wpe)
    wte = init_matrix(tokens.vocab_size, sizes.n_embd)
    wpe = init_matrix(sizes.block_size, sizes.n_embd)

    def lookup(token_id: int, pos_id: int):
        # место под новый эмбеддинг (вектор токена + позиции)
        token_vec = wte[token_id]
        pos_vec = wpe[pos_id % sizes.block_size]
        return [t + p for t, p in zip(token_vec, pos_vec)]

    return Embedding(
        state={"wte": wte, "wpe": wpe},
        lookup=lookup,
        name="learned",
    )
