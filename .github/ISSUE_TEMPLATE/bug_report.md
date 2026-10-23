---
name: Ошибка / баг
about: Программа падает, не ставится окружение, плохой результат
title: ''
labels: bug
assignees: ''
---

## Что произошло

<!-- Опиши: что нажал, что ожидал, что случилось. Приложи текст ошибки целиком. -->

## Логи (обязательно)

Приложи содержимое файлов:

- `logs/bootstrap.log` — установка окружения
- `logs/bootstrap_error.log` — ошибки запуска `main.py` (если создавался)
- `logs/localasr.log` — работа приложения

<details>
<summary>Вывод `doctor`</summary>

```
.venv\Scripts\python run_app.py doctor     # Windows
.venv/bin/python run_app.py doctor         # Linux/macOS
```

</details>

## Окружение

- ОС: <!-- Windows 11 23H2 / Ubuntu 24.04 / macOS 15 -->
- Python: <!-- 3.12.7 -->
- Версия LocalASR: <!-- 1.0.0 -->
- Модель и движок: <!-- ru-turbo-corriolon-int8 / faster-whisper -->
- Железо: <!-- CPU, RAM, GPU -->

## Воспроизведение

1.
2.
3.
