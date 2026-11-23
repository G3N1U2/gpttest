"""Реестр модулей эмбеддинга (блок ЭМБЕДИНГ). Новые PR добавляются сюда."""

from typing import Callable
from structs import Embedding, Sizes, Tokenized
from .learned import embed as embed_learned
from .sinusoidal import embed as embed_sinusoidal
from .rope import embed as embed_rope

EmbeddingFn = Callable[[Tokenized, Sizes], Embedding]

EMBEDDING_REGISTRY: dict[str, EmbeddingFn] = {
    "learned": embed_learned,
    "sinusoidal": embed_sinusoidal,
    "rope": embed_rope,
}


def get_embedding(name: str) -> EmbeddingFn:
    if name not in EMBEDDING_REGISTRY:
        available = ", ".join(sorted(EMBEDDING_REGISTRY.keys()))
        raise KeyError(f"Эмбеддинг '{name}' не найден. Доступны: {available}")
    return EMBEDDING_REGISTRY[name]


__all__ = [
    "EMBEDDING_REGISTRY",
    "get_embedding",
    "embed_learned",
    "embed_sinusoidal",
    "embed_rope",
]
