import json
import math
import os
import random

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

INPUT_PATH = "input.txt"
TEST_RATIO = 0.1
SIZES = Sizes()
TRAIN = TrainConfig()


class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def sample(path: str = INPUT_PATH) -> Dataset:
    docs = read_rows(path)
    shuffle(docs)
    return Dataset(docs=docs)


def split(tokens: Tokenized, ratio: float = TEST_RATIO) -> Split:
    index = split_index(len(tokens.sequences), ratio)
    train_sequences = take_span(tokens.sequences, 0, index)
    test_sequences = take_span(tokens.sequences, index, len(tokens.sequences))
    train = Tokenized(vocab=tokens.vocab, bos=tokens.bos, sequences=train_sequences)
    test = Tokenized(vocab=tokens.vocab, bos=tokens.bos, sequences=test_sequences)
    return Split(train=train, test=test)


def tokenize(data: Dataset) -> Tokenized:
    vocab = build_vocab(data.docs)
    bos = make_bos(vocab)
    sequences = encode_all(data.docs, vocab, bos)
    return Tokenized(vocab=vocab, bos=bos, sequences=sequences)


def embed(tokens: Tokenized, sizes: Sizes = SIZES) -> Embedding:
    wte = init_matrix(tokens.vocab_size, sizes.n_embd)
    wpe = init_matrix(sizes.block_size, sizes.n_embd)
    return Embedding(wte=wte, wpe=wpe)


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
    return Metrics(loss=loss, samples=samples)


def read_rows(path):
    if not os.path.exists(path):
        write_rows(path, download_rows())
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_rows(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def download_rows():
    rows = try_datasets()
    if rows:
        return rows
    return fetch_rows()


def try_datasets():
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    dataset = load_dataset("kuznetsoffandrey/sberquad", split="train[:1000]")
    return to_rows(dataset)


def fetch_rows():
    import urllib.request

    rows = []
    for offset in range(0, 1000, 100):
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset=kuznetsoffandrey%2Fsberquad&config=sberquad&split=train&offset={offset}&length=100"
        )
        with urllib.request.urlopen(url) as response:
            payload = json.loads(response.read().decode("utf-8"))
            rows.extend(to_row(r["row"]) for r in payload.get("rows", []))
    return [row for row in rows if row]


def to_rows(records):
    return [row for row in (to_row(record) for record in records) if row]


def to_row(record):
    question = record.get("question", "").replace("\n", " ").strip()
    answers = record.get("answers", {}).get("text", [])
    answer = answers[0].replace("\n", " ").strip() if answers else ""
    if question and answer:
        return f"{question} {answer}"
    return ""


def shuffle(items):
    random.shuffle(items)


def split_index(count, ratio):
    return max(1, int(count * (1 - ratio)))


def take_span(items, start, end):
    return items[start:end]


def build_vocab(docs):
    return sorted(set("".join(docs)))


def make_bos(vocab):
    return len(vocab)


def encode_all(docs, vocab, bos):
    index = encode_map(vocab)
    return [encode_doc(doc, index, bos) for doc in docs]


def encode_map(vocab):
    return {char: position for position, char in enumerate(vocab)}


def encode_doc(doc, index, bos):
    return [bos] + [index[char] for char in doc] + [bos]


def init_matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def embed_token(embedding, token_id, pos_id):
    token = embedding.wte[token_id]
    position = embedding.wpe[pos_id]
    return [t + p for t, p in zip(token, position)]


def init_state(embedding, vocab_size, sizes):
    state = {
        "wte": embedding.wte,
        "wpe": embedding.wpe,
        "lm_head": init_matrix(vocab_size, sizes.n_embd),
    }
    for index in range(sizes.n_layer):
        add_layer(state, sizes, index)
    return state


def add_layer(state, sizes, index):
    width = sizes.n_embd
    state[f"layer{index}.attn_wq"] = init_matrix(width, width)
    state[f"layer{index}.attn_wk"] = init_matrix(width, width)
    state[f"layer{index}.attn_wv"] = init_matrix(width, width)
    state[f"layer{index}.attn_wo"] = init_matrix(width, width)
    state[f"layer{index}.mlp_fc1"] = init_matrix(4 * width, width)
    state[f"layer{index}.mlp_fc2"] = init_matrix(width, 4 * width)


def flatten_state(state):
    return [param for matrix in state.values() for row in matrix for param in row]


def make_cache(sizes):
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


def forward_layers(model, x, cache):
    for index in range(model.sizes.n_layer):
        x = layer_attention(model, x, cache, index)
        x = layer_mlp(model, x, index)
    return x


def layer_attention(model, x, cache, index):
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


def multi_head(q, keys, values, sizes):
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


def layer_mlp(model, x, index):
    residual = x
    x = rmsnorm(x)
    x = linear(x, model.state[f"layer{index}.mlp_fc1"])
    x = [value.relu() for value in x]
    x = linear(x, model.state[f"layer{index}.mlp_fc2"])
    return [a + b for a, b in zip(x, residual)]


def project(model, x):
    return linear(x, model.state["lm_head"])


def make_batches(sequences, config):
    return [sequences[index % len(sequences)] for index in range(config.steps)]


def init_optimizer(params):
    return {"m": [0.0] * len(params), "v": [0.0] * len(params)}


def run_steps(model, batches, optimizer, config):
    for step, sequence in enumerate(batches):
        loss = sequence_loss(model, sequence)
        loss.backward()
        update_parameters(model.params, optimizer, step, config)
        report_step(step, loss, config)
    return model


def sequence_loss(model, sequence):
    cache = make_cache(model.sizes)
    n = min(model.sizes.block_size, len(sequence) - 1)
    losses = [token_loss(model, cache, sequence, position) for position in range(n)]
    return sum(losses) / n


def token_loss(model, cache, sequence, position):
    logits = predict(model, sequence[position], position, cache)
    probs = softmax(logits)
    return -probs[sequence[position + 1]].log()


def update_parameters(params, optimizer, step, config):
    rate = decayed_rate(config, step)
    for index, param in enumerate(params):
        adjust_param(param, optimizer, index, step, config, rate)


def decayed_rate(config, step):
    return config.lr * (1 - step / config.steps)


def adjust_param(param, optimizer, index, step, config, rate):
    momentum = config.beta1 * optimizer["m"][index] + (1 - config.beta1) * param.grad
    variance = config.beta2 * optimizer["v"][index] + (1 - config.beta2) * param.grad**2
    optimizer["m"][index] = momentum
    optimizer["v"][index] = variance
    momentum_hat = momentum / (1 - config.beta1 ** (step + 1))
    variance_hat = variance / (1 - config.beta2 ** (step + 1))
    param.data -= rate * momentum_hat / (variance_hat**0.5 + config.eps)
    param.grad = 0


def report_step(step, loss, config):
    print(f"step {step + 1:4d} / {config.steps:4d} | loss {loss.data:.4f}", end="\r")


def evaluate(model, tokens, config):
    docs = take_span(tokens.sequences, 0, config.eval_limit)
    return [sequence_loss(model, doc).data for doc in docs]


def average(values):
    return sum(values) / len(values) if values else 0.0


def generate(model, tokens, config):
    return [sample_text(model, tokens, config) for _ in range(config.samples)]


def sample_text(model, tokens, config):
    cache = make_cache(model.sizes)
    token_id = tokens.bos
    chars = []
    for position in range(model.sizes.block_size):
        token_id = next_token(model, tokens, cache, token_id, position, config.temperature)
        if token_id == tokens.bos:
            break
        chars.append(tokens.vocab[token_id])
    return "".join(chars)


def next_token(model, tokens, cache, token_id, position, temperature):
    logits = predict(model, token_id, position, cache)
    probs = softmax([value / temperature for value in logits])
    return random.choices(range(tokens.vocab_size), weights=[p.data for p in probs])[0]


def set_seed(seed):
    random.seed(seed)
