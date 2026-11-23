# gpttest

MicroGPT на чистом Python (только стандартная библиотека, без PyTorch/NumPy), разложенный на этапы пайплайна с модульной подменой токенизаторов и эмбеддингов через `import` и автоматическими тестами в GitHub Actions.

## Пайплайн (правило calls-only)

```text
sample → tokenize → split → embed → build_gpt → predict → train → test
```

`pipeline()` и функции-этапы в `stages.py` состоят только из вызовов — это строго проверяется AST-тестом `tests/test_structure.py`.

## Подмена токенизатора или эмбеддинга через `import`

В `pipeline.py` достаточно заменить две строки импорта:

```python
from tokenizers.bpe import tokenize as default_tokenize
from embeddings.rope import embed as default_embed
```

Либо запустить нужную связку из терминала:

```bash
python main.py --tokenizer bpe --embedding rope --steps 200
```

## Структура репозитория

| Файл / папка | Назначение |
| --- | --- |
| `main.py` | Точка входа CLI: запускает `pipeline()`, печатает `test loss`, `perplexity` и сэмплы |
| `pipeline.py` | `pipeline()` — только вызовы этапов + импорт активного токенизатора и эмбеддинга |
| `stages.py` | Этапы пайплайна (calls-only) + примитивы GPT, Adam и инференса |
| `structs.py` | `Sizes`, `Dataset`, `Tokenized`, `Split`, `Embedding`, `GPT`, `Cache`, `TrainConfig`, `Metrics` |
| `autograd.py` | Скалярный `Value` и `init_matrix` |
| `tokenizers/` | Сменные модули токенизации (`char.py`, `bpe.py`, `word.py`, + новые PR команды) |
| `embeddings/` | Сменные модули эмбеддинга (`learned.py`, `sinusoidal.py`, `rope.py`, + новые PR команды) |
| `tests/test_structure.py` | AST-тест правила calls-only и проверка контрактов модулей |
| `tests/test_pipeline.py` | Runtime-тесты: контракты, размерности `n_embd`, `backward()`, обучение и `test()` |
| `.github/workflows/ci.yml` | GitHub Actions: `compileall`, `test_structure`, матрица `test_pipeline` и отчёт в PR |

## Как участнику команды добавить свой вариант через PR

1. Создайте ветку: `git checkout -b feat/my-tokenizer`
2. Добавьте новый файл в `tokenizers/my_tok.py` (с функцией `tokenize(data: Dataset) -> Tokenized`) или в `embeddings/my_emb.py` (с функцией `embed(tokens: Tokenized, sizes: Sizes) -> Embedding`).
3. Зарегистрируйте его в `tokenizers/__init__.py` или `embeddings/__init__.py`.
4. Проверьте локально:
   ```bash
   python -m compileall -q .
   python -m unittest discover -s tests -v
   ```
5. Откройте Pull Request — GitHub Actions автоматически проверит AST, прогонит тесты и опубликует таблицу метрик в PR.
