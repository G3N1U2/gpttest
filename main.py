import json
import math
import os
import random
import sys

random.seed(42)

INPUT_PATH = "input.txt"

n_layer = 1
n_embd = 16
block_size = 16
n_head = 4
head_dim = n_embd // n_head
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps = 1000
temperature = 0.5

docs = []
uchars = []
BOS = 0
vocab_size = 0
sequences = []
wte = []
wpe = []
state_dict = {}
params = []
m = []
v = []


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

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1
        for node in reversed(topo):
            for child, local_grad in zip(node._children, node._local_grads):
                child.grad += local_grad * node.grad


matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
]


def pipeline(source=INPUT_PATH):
    sample(source)
    tokenize()
    embed()
    build_gpt()
    _ = predict(BOS, 0, *make_cache())
    train()
    test()


def sample(path):
    load_docs(path)
    shuffle_docs()
    report_docs()


def tokenize():
    build_vocab()
    encode_docs()


def embed():
    init_embeddings()


def build_gpt():
    init_state()
    flatten_params()


def predict(token_id, pos_id, keys, values):
    return gpt(token_id, pos_id, keys, values)


def train():
    init_optimizer()
    run_steps()


def test():
    run_inference()


def load_docs(path):
    global docs
    docs = read_rows(path)


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
            rows.extend(to_row(record["row"]) for record in payload.get("rows", []))
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


def shuffle_docs():
    random.shuffle(docs)


def report_docs():
    print(f"num docs: {len(docs)}")


def build_vocab():
    global uchars, BOS, vocab_size
    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"vocab size: {vocab_size}")


def encode_docs():
    global sequences
    index = build_index()
    sequences = [encode_doc(doc, index) for doc in docs]


def build_index():
    return {char: position for position, char in enumerate(uchars)}


def encode_doc(doc, index):
    return [BOS] + [index[char] for char in doc] + [BOS]



def init_embeddings():
    global wte, wpe
    wte = matrix(vocab_size, n_embd)
    wpe = matrix(block_size, n_embd)


def init_state():
    global state_dict
    state_dict = {
        "wte": wte,
        "wpe": wpe,
        "lm_head": matrix(vocab_size, n_embd),
    }
    for index in range(n_layer):
        add_layer(index)


def add_layer(index):
    state_dict[f"layer{index}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{index}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{index}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{index}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{index}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{index}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)


def flatten_params():
    global params
    params = [param for matrix in state_dict.values() for row in matrix for param in row]
    print(f"num params: {len(params)}")


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_residual)]

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict["lm_head"])
    return logits


def make_cache():
    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    return keys, values


def init_optimizer():
    global m, v
    m = [0.0] * len(params)
    v = [0.0] * len(params)


def run_steps():
    for step in range(num_steps):
        sequence = sequences[step % len(sequences)]
        loss = sequence_loss(sequence)
        loss.backward()
        update_params(step)
        print(f"step {step + 1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end="\r")


def sequence_loss(sequence):
    keys, values = make_cache()
    n = min(block_size, len(sequence) - 1)
    losses = [token_loss(sequence, keys, values, pos_id) for pos_id in range(n)]
    return (1 / n) * sum(losses)


def token_loss(sequence, keys, values, pos_id):
    logits = gpt(sequence[pos_id], pos_id, keys, values)
    probs = softmax(logits)
    return -probs[sequence[pos_id + 1]].log()


def update_params(step):
    lr_t = learning_rate * (1 - step / num_steps)
    for index, param in enumerate(params):
        update_param(param, index, step, lr_t)


def update_param(param, index, step, lr_t):
    m[index] = beta1 * m[index] + (1 - beta1) * param.grad
    v[index] = beta2 * v[index] + (1 - beta2) * param.grad**2
    m_hat = m[index] / (1 - beta1 ** (step + 1))
    v_hat = v[index] / (1 - beta2 ** (step + 1))
    param.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
    param.grad = 0


def run_inference():
    print("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(20):
        keys, values = make_cache()
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([value / temperature for value in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"sample {sample_idx + 1:2d}: {''.join(sample)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pipeline()
