"""Точка входа: запускает pipeline() и печатает метрики и сэмплы."""

import argparse
import json
import math
import sys

from embeddings import get_embedding
from pipeline import default_embed, default_tokenize, pipeline
from stages import INPUT_PATH, OFFLINE_SOURCE
from structs import TrainConfig
from tokenizers import get_tokenizer


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="MicroGPT на чистом Python (calls-only pipeline)")
    parser.add_argument("--tokenizer", default=None, help="Ключ токенизатора из tokenizers/")
    parser.add_argument("--embedding", default=None, help="Ключ эмбеддинга из embeddings/")
    parser.add_argument("--steps", type=int, default=1000, help="Число шагов обучения")
    parser.add_argument("--samples", type=int, default=20, help="Количество примеров инференса")
    parser.add_argument("--offline", action="store_true", help="Использовать встроенные примеры SberQuAD без сети")
    parser.add_argument("--json-out", default=None, help="Сохранить отчет метрик в JSON (для GitHub Actions)")
    args = parser.parse_args()

    tokenize_fn = get_tokenizer(args.tokenizer) if args.tokenizer else default_tokenize
    embed_fn = get_embedding(args.embedding) if args.embedding else default_embed
    source = OFFLINE_SOURCE if args.offline else INPUT_PATH
    config = TrainConfig(steps=args.steps, samples=args.samples)

    metrics = pipeline(
        source=source,
        config=config,
        tokenize_fn=tokenize_fn,
        embed_fn=embed_fn,
    )

    perplexity = math.exp(min(20.0, metrics.loss))
    print(f"\ntest loss: {metrics.loss:.4f} | perplexity: {perplexity:.2f}")
    for index, text in enumerate(metrics.samples, 1):
        print(f"sample {index:2d}: {text}")

    if args.json_out:
        report = {
            "tokenizer": args.tokenizer or "bpe",
            "embedding": args.embedding or "rope",
            "vocab_size": metrics.vocab_size,
            "num_params": metrics.num_params,
            "test_loss": round(metrics.loss, 4),
            "perplexity": round(perplexity, 2),
            "passed": math.isfinite(metrics.loss),
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
