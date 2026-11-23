"""
Этапы пайплайна (строго calls-only) и чистые примитивы MicroGPT без сторонних библиотек.
Токенизация и эмбеддинги вынесены в отдельные пакеты tokenizers/ и embeddings/.
"""

import json
import os
import random

from autograd import Value, init_matrix
from embeddings.rope import embed as active_embed
from structs import (
    Cache,
    Dataset,
    Embedding,
    GPT,
    Metrics,
    Sizes,
    Split,
    Tokenized,
    TrainConfig,
)
from tokenizers.bpe import tokenize as active_tokenize

INPUT_PATH = "input.txt"
OFFLINE_SOURCE = "__offline__"
TEST_RATIO = 0.1
SIZES = Sizes()
TRAIN = TrainConfig()

FALLBACK_SBERQUAD_DOCS = [
    "В каком году была основана Москва? В 1147 году",
    "Кто написал роман Война и мир? Лев Толстой",
    "Какая река является самой длинной в Европе? Волга",
    "Что изучает астрономия? Небесные тела и Вселенную",
    "Сколько планет в Солнечной системе? Восемь планет",
    "Кто создал периодическую таблицу элементов? Дмитрий Менделеев",
    "Какой газ преобладает в атмосфере Земли? Азот",
    "Где находится озеро Байкал? В Восточной Сибири",
    "Когда состоялся первый полет человека в космос? 12 апреля 1961 года",
    "Какой город является столицей России? Москва",
    "Что такое фотосинтез? Преобразование света в энергию",
    "Кто изобрел радио в России? Александр Попов",
]


# === Этапы пайплайна (проверяются AST-тестом calls-only) ===

def sample(path: str = INPUT_PATH) -> Dataset:
    docs = read_rows(path)
    shuffle(docs)
    return Dataset(docs=docs)


def tokenize(data: Dataset) -> Tokenized:
    return active_tokenize(data)


def split(tokens: Tokenized, ratio: float = TEST_RATIO) -> Split:
    index = split_index(len(tokens.sequences), ratio)
    train_sequences = take_span(tokens.sequences, 0, index)
    test_sequences = take_span(tokens.sequences, index, len(tokens.sequences))
    train_part = make_tokenized_subset(tokens, train_sequences)
    test_part = make_tokenized_subset(tokens, test_sequences)
    return Split(train=train_part, test=test_part)


def embed(tokens: Tokenized, sizes: Sizes = SIZES) -> Embedding:
    return active_embed(tokens, sizes)


def build_gpt(embedding: Embedding, tokens: Tokenized, sizes: Sizes = SIZES) -> GPT:
    state = init_state(embedding, tokens.vocab_size, sizes)
    params = flatten_state(state)
    return GPT(sizes=sizes, embedding=embedding, state=state, params=params)


def predict(model: GPT, token_id: int, pos_id: int, cache: Cache):
    x = embed_token(model.embedding, token_id, pos_id)
    x = rmsnorm(x)
    x = forward_layers(model, x, cache)
    return project(model, x)


def train(model: GPT, tokens: Tokenized, config: TrainConfig = TRAIN) -> GPT:
    batches = make_batches(tokens.sequences, config)
    optimizer = init_optimizer(model.params)
    model = run_steps(model, batches, optimizer, config)
    return model


def test(model: GPT, tokens: Tokenized, config: TrainConfig = TRAIN) -> Metrics:
    losses = evaluate(model, tokens, config)
    loss = average(losses)
    samples = generate(model, tokens, config)
    return make_metrics(loss, samples, tokens, model)


# === Примитивы (реализация этапов на чистом Python) ===

def read_rows(path: str) -> list[str]:
    if path == OFFLINE_SOURCE:
        return list(FALLBACK_SBERQUAD_DOCS)
    if not os.path.exists(path):
        rows = download_rows()
        if rows:
            write_rows(path, rows)
        else:
            return list(FALLBACK_SBERQUAD_DOCS)
    with open(path, encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return docs if docs else list(FALLBACK_SBERQUAD_DOCS)


def write_rows(path: str, rows: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def download_rows() -> list[str]:
    rows = try_datasets()
    if rows:
        return rows
    return fetch_rows()


def try_datasets() -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    try:
        dataset = load_dataset("kuznetsoffandrey/sberquad", split="train[:1000]")
        return to_rows(dataset)
    except Exception:
        return []


def fetch_rows() -> list[str]:
    import urllib.request

    rows: list[str] = []
    try:
        for offset in range(0, 1000, 100):
            url = (
                "https://datasets-server.huggingface.co/rows"
                f"?dataset=kuznetsoffandrey%2Fsberquad&config=sberquad&split=train&offset={offset}&length=100"
            )
            with urllib.request.urlopen(url, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
                rows.extend(to_row(r["row"]) for r in payload.get("rows", []))
    except Exception:
        return []
    return [row for row in rows if row]


def to_rows(records) -> list[str]:
    return [row for row in (to_row(record) for record in records) if row]


def to_row(record) -> str:
    question = record.get("question", "").replace("\n", " ").strip()
    answers = record.get("answers", {}).get("text", [])
    answer = answers[0].replace("\n", " ").strip() if answers else ""
    if question and answer:
        return f"{question} {answer}"
    return ""


def shuffle(items: list) -> None:
    random.shuffle(items)


def split_index(count: int, ratio: float) -> int:
    if count <= 1:
        return count
    return max(1, min(count - 1, int(count * (1 - ratio))))


def take_span(items: list, start: int, end: int) -> list:
    return items[start:end]


def make_tokenized_subset(tokens: Tokenized, sequences: list[list[int]]) -> Tokenized:
    return Tokenized(
        vocab=tokens.vocab,
        bos=tokens.bos,
        sequences=sequences,
        decode_token=tokens.decode_token,
    )


def embed_token(embedding: Embedding, token_id: int, pos_id: int):
    return embedding.lookup(token_id, pos_id)


def init_state(embedding: Embedding, vocab_size: int, sizes: Sizes) -> dict:
    state = {
        **embedding.state,
        "lm_head": init_matrix(vocab_size, sizes.n_embd),
    }
    for index in range(sizes.n_layer):
        add_layer(state, sizes, index)
    return state


def add_layer(state: dict, sizes: Sizes, index: int) -> None:
    width = sizes.n_embd
    state[f"layer{index}.attn_wq"] = init_matrix(width, width)
    state[f"layer{index}.attn_wk"] = init_matrix(width, width)
    state[f"layer{index}.attn_wv"] = init_matrix(width, width)
    state[f"layer{index}.attn_wo"] = init_matrix(width, width)
    state[f"layer{index}.mlp_fc1"] = init_matrix(4 * width, width)
    state[f"layer{index}.mlp_fc2"] = init_matrix(width, 4 * width)


def flatten_state(state: dict) -> list:
    return [param for matrix in state.values() for row in matrix for param in row]


def make_cache(sizes: Sizes) -> Cache:
    keys = [[] for _ in range(sizes.n_layer)]
    values = [[] for _ in range(sizes.n_layer)]
    return Cache(keys=keys, values=values)


def linear(x, w):
    return [sum(weight * value for weight, value in zip(row, x)) for row in w]


def softmax(logits):
    peak = max(value.data for value in logits)
    exps = [(value - peak).exp() for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def rmsnorm(x):
    mean_square = sum(value * value for value in x) / len(x)
    scale = (mean_square + 1e-5) ** -0.5
    return [value * scale for value in x]


def forward_layers(model: GPT, x, cache: Cache):
    for index in range(model.sizes.n_layer):
        x = layer_attention(model, x, cache, index)
        x = layer_mlp(model, x, index)
    return x


def layer_attention(model: GPT, x, cache: Cache, index: int):
    residual = x
    x = rmsnorm(x)
    q = linear(x, model.state[f"layer{index}.attn_wq"])
    k = linear(x, model.state[f"layer{index}.attn_wk"])
    v = linear(x, model.state[f"layer{index}.attn_wv"])
    cache.keys[index].append(k)
    cache.values[index].append(v)
    attended = multi_head(q, cache.keys[index], cache.values[index], model.sizes)
    output = linear(attended, model.state[f"layer{index}.attn_wo"])
    return [a + b for a, b in zip(output, residual)]


def multi_head(q, keys, values, sizes: Sizes):
    outputs = []
    for head in range(sizes.n_head):
        start = head * sizes.head_dim
        stop = start + sizes.head_dim
        q_head = q[start:stop]
        k_head = [k[start:stop] for k in keys]
        v_head = [v[start:stop] for v in values]
        weights = softmax(attention_logits(q_head, k_head))
        outputs.extend(weighted_sum(weights, v_head))
    return outputs


def attention_logits(q_head, k_head):
    return [sum(a * b for a, b in zip(q_head, k)) / len(q_head) ** 0.5 for k in k_head]


def weighted_sum(weights, vectors):
    return [
        sum(weight * vector[position] for weight, vector in zip(weights, vectors))
        for position in range(len(vectors[0]))
    ]


def layer_mlp(model: GPT, x, index: int):
    residual = x
    x = rmsnorm(x)
    x = linear(x, model.state[f"layer{index}.mlp_fc1"])
    x = [value.relu() for value in x]
    x = linear(x, model.state[f"layer{index}.mlp_fc2"])
    return [a + b for a, b in zip(x, residual)]


def project(model: GPT, x):
    return linear(x, model.state["lm_head"])


def make_batches(sequences: list[list[int]], config: TrainConfig) -> list[list[int]]:
    return [sequences[index % len(sequences)] for index in range(config.steps)]


def init_optimizer(params: list) -> dict:
    return {"m": [0.0] * len(params), "v": [0.0] * len(params)}


def run_steps(model: GPT, batches: list[list[int]], optimizer: dict, config: TrainConfig) -> GPT:
    for step, sequence in enumerate(batches):
        loss = sequence_loss(model, sequence)
        loss.backward()
        update_parameters(model.params, optimizer, step, config)
        report_step(step, loss, config)
    return model


def sequence_loss(model: GPT, sequence: list[int]):
    cache = make_cache(model.sizes)
    n = min(model.sizes.block_size, len(sequence) - 1)
    if n <= 0:
        return Value(0.0)
    losses = [token_loss(model, cache, sequence, position) for position in range(n)]
    return sum(losses) / n


def token_loss(model: GPT, cache: Cache, sequence: list[int], position: int):
    logits = predict(model, sequence[position], position, cache)
    probs = softmax(logits)
    return -probs[sequence[position + 1]].log()


def update_parameters(params: list, optimizer: dict, step: int, config: TrainConfig) -> None:
    rate = decayed_rate(config, step)
    for index, param in enumerate(params):
        adjust_param(param, optimizer, index, step, config, rate)


def decayed_rate(config: TrainConfig, step: int) -> float:
    return config.lr * (1 - step / max(1, config.steps))


def adjust_param(param, optimizer: dict, index: int, step: int, config: TrainConfig, rate: float) -> None:
    momentum = config.beta1 * optimizer["m"][index] + (1 - config.beta1) * param.grad
    variance = config.beta2 * optimizer["v"][index] + (1 - config.beta2) * param.grad**2
    optimizer["m"][index] = momentum
    optimizer["v"][index] = variance
    momentum_hat = momentum / (1 - config.beta1 ** (step + 1))
    variance_hat = variance / (1 - config.beta2 ** (step + 1))
    param.data -= rate * momentum_hat / (variance_hat**0.5 + config.eps)
    param.grad = 0


def report_step(step: int, loss: Value, config: TrainConfig) -> None:
    print(f"step {step + 1:4d} / {config.steps:4d} | loss {loss.data:.4f}", end="\r")


def evaluate(model: GPT, tokens: Tokenized, config: TrainConfig) -> list[float]:
    docs = take_span(tokens.sequences, 0, config.eval_limit)
    return [sequence_loss(model, doc).data for doc in docs]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def generate(model: GPT, tokens: Tokenized, config: TrainConfig) -> list[str]:
    return [sample_text(model, tokens, config) for _ in range(config.samples)]


def sample_text(model: GPT, tokens: Tokenized, config: TrainConfig) -> str:
    cache = make_cache(model.sizes)
    token_id = tokens.bos
    pieces: list[str] = []
    for position in range(model.sizes.block_size):
        token_id = next_token(model, tokens, cache, token_id, position, config.temperature)
        if token_id == tokens.bos:
            break
        pieces.append(tokens.decode_token(token_id))
    return "".join(pieces).strip()


def next_token(model: GPT, tokens: Tokenized, cache: Cache, token_id: int, position: int, temperature: float) -> int:
    logits = predict(model, token_id, position, cache)
    probs = softmax([value / temperature for value in logits])
    return random.choices(range(tokens.vocab_size), weights=[p.data for p in probs])[0]


def make_metrics(loss: float, samples: list[str], tokens: Tokenized, model: GPT) -> Metrics:
    return Metrics(
        loss=loss,
        samples=samples,
        vocab_size=tokens.vocab_size,
        num_params=len(model.params),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
