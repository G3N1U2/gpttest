## Новый модуль для gpttest (Pure Python)

### Что добавлено
- [ ] Новый токенизатор в `tokenizers/<name>.py` (функция `tokenize(data: Dataset) -> Tokenized`)
- [ ] Новый эмбеддинг в `embeddings/<name>.py` (функция `embed(tokens: Tokenized, sizes: Sizes) -> Embedding`)
- [ ] Изменения в `stages.py` / `pipeline.py`

### Как подключить через import
```python
from tokenizers.bpe import tokenize
from embeddings.rope import embed
```

### Чек-лист перед отправкой PR
- [ ] Только стандартная библиотека Python (без PyTorch / NumPy)
- [ ] Новый модуль зарегистрирован в `tokenizers/__init__.py` или `embeddings/__init__.py`
- [ ] Пройдена проверка синтаксиса: `python -m compileall -q .`
- [ ] Пройдены AST-тесты и runtime-тесты: `python -m unittest discover -s tests -v`
