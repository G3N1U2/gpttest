"""
Функциональные runtime-тесты блока «ТЕСТ» на встроенном unittest (чистый Python).
Реально запускает tokenize -> split -> embed -> build_gpt -> predict -> train -> test
без выхода в сеть, проверяя размерности, градиенты и сходимость обучения.
"""

import math
import os
import unittest

from autograd import Value
from embeddings import EMBEDDING_REGISTRY, get_embedding
from pipeline import pipeline
from stages import (
    OFFLINE_SOURCE,
    SIZES,
    build_gpt,
    make_cache,
    predict,
    sample,
    sequence_loss,
    set_seed,
    split,
    train,
)
from structs import TrainConfig
from tokenizers import TOKENIZER_REGISTRY, get_tokenizer


class MicroGPTPipelineRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tok_name = os.getenv("MICROGPT_TOKENIZER", "bpe")
        self.emb_name = os.getenv("MICROGPT_EMBEDDING", "rope")
        self.tokenize_fn = get_tokenizer(self.tok_name)
        self.embed_fn = get_embedding(self.emb_name)
        set_seed(42)
        self.data = sample(OFFLINE_SOURCE)

    def test_tokenizer_contract_and_bos_boundaries(self):
        tokens = self.tokenize_fn(self.data)
        self.assertGreater(tokens.vocab_size, 2)
        self.assertEqual(len(tokens.sequences), len(self.data.docs))
        for seq in tokens.sequences:
            self.assertGreaterEqual(len(seq), 3)
            self.assertEqual(seq[0], tokens.bos)
            self.assertEqual(seq[-1], tokens.bos)
            for tid in seq:
                self.assertGreaterEqual(tid, 0)
                self.assertLess(tid, tokens.vocab_size)
        decoded = "".join(tokens.decode_token(tid) for tid in tokens.sequences[0] if tid != tokens.bos)
        self.assertTrue(decoded.strip(), "decode_token должен восстанавливать непустую строку")

    def test_embedding_dimensions_and_gpt_forward_backward(self):
        tokens = self.tokenize_fn(self.data)
        embedding = self.embed_fn(tokens, SIZES)
        self.assertTrue(embedding.state, "Embedding.state должен содержать обучаемые матрицы")

        vec = embedding.lookup(tokens.bos, 0)
        self.assertEqual(
            len(vec),
            SIZES.n_embd,
            f"Ошибка размерности Матр. эмб.: ожидалось {SIZES.n_embd}, получено {len(vec)}",
        )
        self.assertTrue(all(isinstance(v, Value) for v in vec))

        model = build_gpt(embedding, tokens, SIZES)
        logits = predict(model, tokens.bos, 0, make_cache(SIZES))
        self.assertEqual(len(logits), tokens.vocab_size)

        loss = sequence_loss(model, tokens.sequences[0])
        self.assertTrue(math.isfinite(loss.data))
        loss.backward()
        has_nonzero_grad = any(abs(p.grad) > 0 for p in model.params)
        self.assertTrue(has_nonzero_grad, "Градиенты после backward() не должны быть нулевыми")

    def test_end_to_end_pipeline_training_and_eval(self):
        cfg = TrainConfig(steps=10, samples=2, eval_limit=3)
        tokens = self.tokenize_fn(self.data)
        parts = split(tokens)
        embedding = self.embed_fn(tokens, SIZES)
        model = build_gpt(embedding, tokens, SIZES)

        initial_loss = sequence_loss(model, parts.train.sequences[0]).data
        trained_model = train(model, parts.train, cfg)
        final_loss = sequence_loss(trained_model, parts.train.sequences[0]).data

        self.assertTrue(math.isfinite(initial_loss))
        self.assertTrue(math.isfinite(final_loss))
        self.assertLess(
            final_loss,
            initial_loss,
            f"Loss должен снижаться за 10 шагов ({initial_loss:.4f} -> {final_loss:.4f})",
        )

        metrics = pipeline(
            source=OFFLINE_SOURCE,
            config=TrainConfig(steps=6, samples=2, eval_limit=2),
            tokenize_fn=self.tokenize_fn,
            embed_fn=self.embed_fn,
        )
        self.assertTrue(math.isfinite(metrics.loss))
        self.assertEqual(len(metrics.samples), 2)

    def test_all_registered_combinations_initialize_cleanly(self):
        for tok_key, tok_fn in TOKENIZER_REGISTRY.items():
            tokens = tok_fn(self.data)
            for emb_key, emb_fn in EMBEDDING_REGISTRY.items():
                with self.subTest(tokenizer=tok_key, embedding=emb_key):
                    emb = emb_fn(tokens, SIZES)
                    vec = emb.lookup(tokens.bos, 0)
                    self.assertEqual(len(vec), SIZES.n_embd)


if __name__ == "__main__":
    unittest.main()
