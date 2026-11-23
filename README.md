# gpttest

microgpt на чистом Python: обучение и inference GPT без зависимостей.

## Запуск

```bash
python main.py
```

Python 3.10+. Если `input.txt` нет, данные скачаются автоматически.

## CI

При каждом push GitHub Actions проверяет синтаксис: `python -m py_compile main.py`.
