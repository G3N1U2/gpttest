# gpttest — microgpt

Учебный разбор microgpt: пайплайн разбит на этапы, каждый этап — отдельная функция.

## Схема

```text
ОБ ВЫБОРКА ──► ТОКЕН ──► ЭМБЕДДИНГ ──► СТРУКТУРА GPT ──► ПРЕДСКАЗАНИЕ
                                                             │
                                                 ОБУЧАЕМ ◄───┘
                                                    │
                                                  ТЕСТ
```

## Правило

**Функции-этапы состоят только из вызовов.** Ни арифметики, ни циклов, ни ветвлений —
вся реальная работа живёт в примитивах в конце `stages.py`.

Это проверяется автоматически: `tests/test_structure.py` разбирает исходники через `ast`.

## Файлы

| Файл | Что внутри |
|---|---|
| `pipeline.py` | `pipeline(source)` — только вызовы этапов |
| `stages.py` | этапы (только вызовы) + примитивы-заглушки |
| `structs.py` | объекты, которыми обмениваются этапы |
| `tests/test_structure.py` | проверка правила «только вызовы» |
| `123.py` | исходный скрипт microgpt (справочник) |

## Запуск

```bash
python -m py_compile structs.py stages.py pipeline.py   # синтаксис
python -m unittest discover -s tests -v                  # проверка структуры
python pipeline.py                                       # пайплайн (пока доходит до заглушки)
```

Зависимостей нет — только Python 3.10+.

## Что дальше

Заполнять примитивы в `stages.py` по порядку цепочки:

1. данные — `read_lines`, `keep_nonempty`, `take_train`, `take_test`
2. токенизация — `collect_chars`, `build_vocab`, `make_bos`, `encode_all`, `decode`
3. эмбеддинг — `init_matrix`, `lookup`, `lookup_one`, `make_cache`
4. GPT — `init_layers`, `init_head`, `flatten_state`, `forward_layers`, `project`, `linear`, `rmsnorm`, `softmax`
5. обучение — `make_batches`, `run_steps`, `compute_loss`, `backward`, `adam_update`, `zero_grad`
6. тест — `evaluate`, `average`, `generate`

В docstring каждой заглушки есть ссылка вида `123.py:139-141` — где взять готовую логику.
