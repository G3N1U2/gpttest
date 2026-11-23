"""
Главный пайплайн MicroGPT (правило calls-only).
Подмена токенизатора или эмбеддинга от команды — две строки import ниже.
"""

# === Точка подмены модулей команды через import ===
from embeddings.rope import embed as default_embed
from tokenizers.bpe import tokenize as default_tokenize
# ==================================================

from stages import (
    INPUT_PATH,
    SIZES,
    TRAIN,
    build_gpt,
    make_cache,
    predict,
    sample,
    set_seed,
    split,
    test,
    train,
)
from structs import Metrics, Sizes, TrainConfig


def pipeline(
    source: str = INPUT_PATH,
    config: TrainConfig = TRAIN,
    tokenize_fn=default_tokenize,
    embed_fn=default_embed,
    sizes: Sizes = SIZES,
) -> Metrics:
    set_seed(config.seed)
    data = sample(source)
    tokens = tokenize_fn(data)
    parts = split(tokens)
    embedding = embed_fn(tokens, sizes)
    model = build_gpt(embedding, tokens, sizes)
    _ = predict(model, tokens.bos, 0, make_cache(model.sizes))
    model = train(model, parts.train, config)
    metrics = test(model, parts.test, config)
    return metrics
