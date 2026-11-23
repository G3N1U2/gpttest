from stages import (
    INPUT_PATH,
    TRAIN,
    build_gpt,
    embed,
    make_cache,
    predict,
    sample,
    set_seed,
    split,
    test,
    tokenize,
    train,
)
from structs import Metrics, TrainConfig


def pipeline(source: str = INPUT_PATH, config: TrainConfig = TRAIN) -> Metrics:
    set_seed(config.seed)
    data = sample(source)
    tokens = tokenize(data)
    parts = split(tokens)
    embedding = embed(tokens)
    model = build_gpt(embedding, tokens)
    _ = predict(model, tokens.bos, 0, make_cache(model.sizes))
    model = train(model, parts.train, config)
    metrics = test(model, parts.test, config)
    return metrics
