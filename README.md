# gpttest

microgpt на чистом Python, разложенный на этапы пайплайна в одном файле.

## Пайплайн

```text
sample → tokenize → embed → build_gpt → predict → train → test
```

`pipeline()` и функции-этапы состоят только из вызовов. Реальная работа — в примитивах ниже по файлу.

## Алгоритм

Тот же, что в исходном microgpt: `Value` с автоградом, эмбеддинги `wte`/`wpe`, один слой attention + MLP, Adam, 1000 шагов, inference на 20 сэмплов. Ничего не добавлено и не изменено по математике.

## Запуск

```bash
python main.py
```

Python 3.10+. Если `input.txt` нет, данные скачаются автоматически.

## Проверки

```bash
python -m py_compile main.py
python -m unittest discover -s tests -v
```

`tests/test_structure.py` проверяет через `ast`, что функции-этапы состоят только из вызовов.

## CI

GitHub Actions запускает обе проверки при каждом push.
