"""Структуры данных, которыми обмениваются этапы.

Схема (доска):

    ОБ ВЫБОРКА ──► ТОКЕН ──► ЭМБЕДДИНГ ──► СТРУКТУРА GPT ──► ПРЕДСКАЗАНИЕ
                                                                │
                                                    ОБУЧАЕМ ◄───┘
                                                       │
                                                     ТЕСТ

Каждый этап принимает объект предыдущего этапа и возвращает свой.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Класс Value из 123.py пока не перенесён сюда — используем Any как заглушку типа.
Value = Any


@dataclass(frozen=True)
class Sizes:
    """«РАЗМЕРЫ, СТРУКТУРА» — параметры, общие для эмбеддинга и GPT."""

    n_layer: int = 1
    n_embd: int = 16
    block_size: int = 16
    n_head: int = 4

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass(frozen=True)
class Dataset:
    """ОБ ВЫБОРКА: список документов (непустых строк)."""

    docs: list[str]


@dataclass(frozen=True)
class Split:
    """Результат деления выборки на обучающую и тестовую части."""

    train: Dataset
    test: Dataset


@dataclass(frozen=True)
class Tokenized:
    """ТОКЕН: словарь символов и последовательности id («сл. токенов»)."""

    vocab: list[str]
    bos: int
    sequences: list[list[int]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1


@dataclass(frozen=True)
class Embedding:
    """ЭМБЕДДИНГ: «матp эмб» wte/wpe и готовые векторы по последовательностям."""

    wte: list[list[Value]]
    wpe: list[list[Value]]
    vectors: list[list[Value]]


@dataclass
class GPT:
    """СТРУКТУРА GPT: слои внимания/MLP, голова и плоский список параметров."""

    sizes: Sizes
    embedding: Embedding
    layers: dict[str, list[list[Value]]]
    head: list[list[Value]]
    params: list[Value]

    def state_dict(self) -> dict[str, list[list[Value]]]:
        return {**self.layers, "lm_head": self.head}


@dataclass
class Cache:
    """Кэш attention: накопленные keys/values по слоям."""

    keys: list[list[Value]]
    values: list[list[Value]]


@dataclass(frozen=True)
class TrainConfig:
    """Гиперпараметры обучения (значения взяты из 123.py)."""

    num_steps: int = 1000
    learning_rate: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps: float = 1e-8
    seed: int = 42
    temperature: float = 0.5
    n_samples: int = 20


@dataclass(frozen=True)
class Metrics:
    """ТЕСТ: потеря на выборке и сгенерированные примеры."""

    loss: float
    samples: list[str]
