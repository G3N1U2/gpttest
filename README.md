# gpttest

microgpt на чистом Python, разложенный на этапы пайплайна.

## Пайплайн

```text
sample → tokenize → split → embed → build_gpt → predict → train → test
```

`pipeline()` и функции-этапы состоят только из вызовов. Реальная работа — в примитивах `stages.py`.

## Файлы

| Файл | Что внутри |
|---|---|
| `main.py` | точка входа: запускает `pipeline()` и печатает результаты |
| `pipeline.py` | `pipeline(source, config)` — только вызовы этапов |
| `stages.py` | этапы (только вызовы) + примитивы и класс `Value` |
| `structs.py` | `Dataset`, `Tokenized`, `Split`, `Embedding`, `GPT`, `Cache`, `TrainConfig`, `Metrics` |
| `tests/test_structure.py` | проверка правила «только вызовы» |

## Запуск

```bash
python main.py
```

Python 3.10+. Если `input.txt` нет, данные скачаются автоматически.

## Проверки

```bash
python -m py_compile structs.py stages.py pipeline.py main.py
python -m unittest discover -s tests -v
```

## CI

GitHub Actions запускает обе проверки при каждом push.
