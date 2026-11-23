"""Итоговый пайплайн microgpt.

Требование: тело pipeline() состоит ТОЛЬКО из вызовов этапов —
никакой арифметики, ветвлений, циклов и работы с данными.
Вся логика живёт в stages.py (этапы) и примитивах внутри него.

Порядок этапов — как на доске:
    ОБ ВЫБОРКА → ТОКЕН → ЭМБЕДДИНГ → СТРУКТУРА GPT →
    ПРЕДСКАЗАНИЕ → ОБУЧАЕМ (цикл) → ТЕСТ
"""
from __future__ import annotations

from stages import (
    INPUT_PATH,
    build_gpt,
    embed,
    make_cache,
    predict,
    sample,
    split,
    test,
    tokenize,
    train,
)
from structs import Metrics


def pipeline(source: str = INPUT_PATH) -> Metrics:
    """Полный проход: от файла с данными до метрик на тесте."""
    data = sample(source)
    parts = split(data)
    tokens = tokenize(parts.train)
    embedding = embed(tokens)
    model = build_gpt(embedding)
    _ = predict(model, tokens.bos, 0, make_cache(model))
    model = train(model, tokens)
    metrics = test(model, tokenize(parts.test))
    return metrics


if __name__ == "__main__":
    pipeline()
