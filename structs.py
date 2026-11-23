from dataclasses import dataclass


@dataclass(frozen=True)
class Sizes:
    n_layer: int = 1
    n_embd: int = 16
    block_size: int = 16
    n_head: int = 4

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass(frozen=True)
class Dataset:
    docs: list[str]


@dataclass(frozen=True)
class Tokenized:
    vocab: list[str]
    bos: int
    sequences: list[list[int]]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 1


@dataclass(frozen=True)
class Split:
    train: Tokenized
    test: Tokenized


@dataclass(frozen=True)
class Embedding:
    wte: list
    wpe: list


@dataclass
class GPT:
    sizes: Sizes
    embedding: Embedding
    state: dict
    params: list


@dataclass
class Cache:
    keys: list
    values: list


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 1000
    lr: float = 0.01
    beta1: float = 0.85
    beta2: float = 0.99
    eps: float = 1e-8
    seed: int = 42
    temperature: float = 0.5
    samples: int = 20
    eval_limit: int = 20


@dataclass(frozen=True)
class Metrics:
    loss: float
    samples: list[str]
