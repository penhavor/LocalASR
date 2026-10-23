<div align="center">

# 🎙️ LocalASR

**Локальное распознавание речи: аудио и видео → текст. Без облака, без регистрации, без отправки данных.**

PySide6 · faster-whisper (CTranslate2) · whisper.cpp · Vosk · Python 3.10–3.14+

Запускаешь `main.py` → программа сама создаёт `.venv`, сама ставит зависимости,
сама перезапускается **без консоли** → ты выбираешь файлы и модель → получаешь
`лекция 1.txt` рядом с проектом (или куда скажешь).

[Установка](#-быстрый-старт) · [Как выбрать модель](#-какую-модель-выбрать) ·
[Железо](#-требования-к-железу) · [CLI](#-командная-строка-cli) · [Устранение проблем](#-устранение-проблем)

</div>

---

## ✨ Что умеет

| | |
|---|---|
| 📂 **Любые медиа** | `mp4, mkv, mov, avi, webm, mp3, wav, m4a, flac, ogg, opus…` — видео автоматически превращается в звук (ffmpeg в комплекте, системный ставить **не нужно**) |
| 🧠 **Несколько файлов сразу** | Можно выбрать десяток лекций или целую папку; drag & drop тоже работает |
| 🗂 **Каталог моделей** | У каждой модели показаны **размер файла, потребление RAM/VRAM, поддерживаемые языки, оценки точности и скорости** — выбираешь под своё железо и терпение |
| ⬇️ **Установка моделей в один клик** | Скачивается в `models/` внутри проекта; после установки сразу появляется в списке и становится доступной для выбора |
| 🔗 **Любая своя модель** | Вставляешь ссылку (Hugging Face, прямая ссылка на `.bin`/`.zip`) или путь на диске — движок определится автоматически |
| 🇷🇺 **Приоритет русского** | В каталоге и универсальные мультиязычные модели, и специализированные русские fine-tune'ы |
| 🧹 **Анти-галлюцинации** | Вычищаются `[музыка]`, `[аплодисменты]`, «Спасибо за просмотр», субтитрные вставки, зацикленные повторы фраз и мусор на тишине |
| ⚡ **Скорость** | Batched-декодинг, int8-квантование, Silero VAD (режет тишину), prefetched-декодирование следующего файла, модель грузится один раз на всю очередь |
| 📝 **Форматы результата** | `.txt`, `.txt` с таймкодами, `.srt`, `.vtt`, `.json` (сегменты + слова), `.md` |
| 🖥 **GUI без консоли** | PySide6 (не Tkinter), тёмная тема, живой журнал, прогресс с оценкой остатка времени, отмена в любой момент |
| 🔒 **Чистое окружение** | Всё ставится строго в `.venv` внутри папки проекта — системный Python не трогается |

---

## 🚀 Быстрый старт

### Windows

1. Установи [Python 3.10 или новее](https://www.python.org/downloads/windows/) (64-bit).
   Проверено вплоть до **3.14**; более новые версии принимаются автоматически.
   В установщике обязательно отметь **«Add python.exe to PATH»**.
2. Скачай/склонируй этот репозиторий в любую папку **без кириллицы и пробелов**
   (например, `C:\LocalASR`).
3. Дважды кликни **`main.py`**.

Дальше всё само:

```
[LocalASR] Ищу подходящий Python…
[LocalASR] Создаю виртуальное окружение .venv…
[LocalASR] Ставлю зависимости (PySide6, faster-whisper, …)
[LocalASR] Перезапускаю GUI без консоли
```

Первый запуск занимает 2–5 минут (качается ~350 МБ зависимостей). Все последующие —
мгновенно: проверка зависимостей пропускается по «отпечатку» окружения.

> Консоль на время установки **остаётся открытой намеренно** — иначе при ошибке ты
> увидишь только мелькнувшее окно. Как только GUI поднимется, консоль закроется,
> и дальше программа работает без неё (перезапуск через `pythonw.exe`).

### Linux / macOS

```bash
git clone https://github.com/local-asr/LocalASR
cd LocalASR
python3 main.py
```

### Проверка без GUI

```bash
python3 main.py                      # подготовка окружения + GUI
.venv/bin/python run_app.py doctor    # диагностика (Linux/macOS)
.venv\Scripts\python run_app.py doctor  # Windows
```

---

## 🧭 Как пользоваться

```
┌───────────────────────────────────────────┬──────────────────────────────┐
│ 1 · ФАЙЛЫ                                 │  МОДЕЛИ                      │
│  [📂 Выбрать файлы…] [🗀 Папку…] [Очистить]│  [поиск] [все движки ▾]      │
│  ┌──────────────────────────────────────┐ │  ☐ только установленные      │
│  │ 🎬 лекция 1.mp4      1.2 ГБ        ✕ │ │  ☑ с русским языком          │
│  │ 🎧 подкаст.mp3        84 МБ        ✕ │ │  ┌──────────────────────────┐│
│  └──────────────────────────────────────┘ │  │ ★ Whisper L-v3 Turbo      ││
│                                           │  │   Russian (int8, CT2)    ││
│ 2 · КАК РАСПОЗНАВАТЬ И КУДА СОХРАНЯТЬ      │  │   faster-whisper · 781 МБ││
│  Язык: [Русский ▾]  Формат: [Текст ▾]      │  │   🧠 RAM 2 ГБ · VRAM 2.2 ││
│  Куда: [/…/outputs] [Обзор…] [↺]          │  │   🌐 RU, UK, BE, KK, EN  ││
│  ☑ VAD  ☑ Анти-галлюцинации  ☑ Абзацы     │  │   точность ★★★★★          ││
│  ▸ Расширенные настройки                   │  │   [Подробнее] [Скачать]  ││
│                                           │  └──────────────────────────┘│
│ 3 · РАБОТА                                │  ┌──────────────────────────┐│
│  ██████████████░░░░░░ 68%                 │  │ ＋ Своя модель (ссылка)  ││
│  Файл 2 из 3 · 12:04/41:10 · 3.1×         │  └──────────────────────────┘│
│  [        РАБОТАТЬ · 3 файла      ][Отмена]│                              │
└───────────────────────────────────────────┴──────────────────────────────┘
```

1. **Файлы** — «Выбрать файлы…» (можно несколько через Ctrl/Shift), «Папку…»
   (рекурсивно) или просто перетащи их в окно.
2. **Модель** — справа карточки моделей. По умолчанию не установлено ничего:
   программа сама покажет баннер с рекомендацией под твоё железо. Жми «Скачать»,
   дождись окончания — модель появится в списке с пометкой «✓ установлена»,
   выбери её.
3. **Параметры** — язык (по умолчанию русский), формат результата, папка
   сохранения (по умолчанию `outputs/` в проекте), VAD, анти-галлюцинации,
   абзацы; в «Расширенных» — hotwords, initial_prompt, beam size, batched,
   устройство (CPU/CUDA), потоки, параллельная обработка.
4. **РАБОТАТЬ** — кнопка активна, только когда есть и файлы, и установленная
   модель. Прогресс виден по каждому файлу, распознанный текст печатается
   «вживую».

Результат: `outputs/лекция 1.txt` — **имя совпадает с именем оригинала**.
Если файл с таким именем уже есть, создаётся `лекция 1 (2).txt` (ничего не
перетирается), либо включи перезапись в настройках.

---

## 🧠 Какую модель выбрать

Все модели — современные (Whisper **Large v3 Turbo** и его русские fine-tune'ы,
а не «классика 2020 года»). Размеры указаны по фактическим данным Hugging Face.

| Модель | Движок | Файл | RAM | Языки | Точн. | Скор. | Кому |
|---|---|---|---|---|---|---|---|
| **★ Whisper L-v3 Turbo · Russian (int8, CT2)** | faster-whisper | 819 МБ | ~2 ГБ | **RU** + 99 | ★★★★★ | ★★★★ | **основной выбор для русских лекций** |
| ★ Whisper L-v3 Turbo q5_0 | whisper.cpp | 574 МБ | ~1.2 ГБ | **RU** + 99 | ★★★★ | ★★★★ | точность при 4–8 ГБ RAM без GPU |
| ★ Whisper L-v3 Turbo (multilingual, CT2) | faster-whisper | 1.5 ГБ | ~2.6 ГБ | 99 | ★★★★ | ★★★★★ | язык неизвестен / запись многоязычная |
| ★ Vosk Russian Small 0.22 | Vosk | 45 МБ | ~0.4 ГБ | **RU** | ★ | ★★★★★ | совсем слабое железо, черновик |
| Whisper L-v3 Turbo · Russian (float16) | faster-whisper | 1.5 ГБ | ~2.6 ГБ | **RU** + 99 | ★★★★★ | ★★★★ | максимум точности на русском |
| Whisper L-v3 Turbo · Russian **Code-Switch** | faster-whisper | 782 МБ | ~2 ГБ | **RU+EN** | ★★★★★ | ★★★★ | IT-лекции с английскими терминами |
| Whisper L-v3 Turbo · Russian (dvislobokov) | faster-whisper | 1.5 ГБ | ~2.6 ГБ | **RU**, EN | ★★★★ | ★★★★ | альтернатива для сравнения |
| Whisper Large v3 | faster-whisper | 2.9 ГБ | ~4.7 ГБ | 99 | ★★★★★ | ★★ | максимум качества, время не важно |
| Whisper Large v3 q5_0 | whisper.cpp | 1.1 ГБ | ~2 ГБ | 99 | ★★★★ | ★★ | редкие языки при малом RAM |
| Whisper Medium / Small / Base | faster-whisper | 1.5 / 484 / 145 МБ | 2.6 / 1.5 / 0.7 ГБ | 99 | ★★★/★★/★ | ★★★/★★★★/★★★★★ | компромиссы по скорости |
| Whisper L-v3 Turbo (whisper.cpp full) | whisper.cpp | 1.5 ГБ | ~2.6 ГБ | 99 | ★★★★★ | ★★★ | GGML без квантования |
| Vosk Russian 0.42 | Vosk | 1.8 ГБ | ~2.5 ГБ | **RU** | ★★★ | ★★★★★ | быстрый CPU-прогон русской речи |
| Vosk Multilingual 0.27 | Vosk | 1.2 ГБ | ~1.8 ГБ | ~20 | ★★ | ★★★★ | автоопределение языка на слабом ПК |
| Distil Whisper Large v3 | faster-whisper | 1.4 ГБ | ~2.6 ГБ | **EN** | ★★★★ | ★★★★★ | английские доклады |

**Коротко:** русские лекции → `Whisper Large v3 Turbo · Russian (int8)`.
Мало памяти → `Whisper L-v3 Turbo q5_0 (whisper.cpp)`.
Совсем слабый ПК → `Vosk Russian Small`.

### Своя модель

Кнопка **«＋ Своя модель (ссылка / путь)»** понимает:

* `coriollon/whisper-large-v3-turbo-russian` — репозиторий Hugging Face;
* `owner/name/tree/main/ct2_int8_float16` — репозиторий + подпапка;
* `https://…/ggml-large-v3-turbo-q5_0.bin` — прямая ссылка на файл;
* `https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip` — архив Vosk (распакуется сам);
* `D:\models\my-whisper` — папка на диске.

Движок определяется автоматически по содержимому: `model.bin` + `config.json`
→ faster-whisper, `ggml-*.bin` → whisper.cpp, `am/` + `graph/` → Vosk.

---

## 🖥 Требования к железу

| Что | Минимум | Комфортно |
|---|---|---|
| ОС | Windows 10/11, Linux, macOS | — |
| Python | 3.10 (64-bit) | 3.11–3.14; новее — тоже работает |
| RAM | 4 ГБ (модели до 500 МБ) | 8–16 ГБ (turbo int8) |
| Диск | 3 ГБ под проект + модели | 10 ГБ (несколько моделей) |
| GPU | не нужен | NVIDIA с 6+ ГБ VRAM и CUDA 12 |

Программа сама определяет CPU/RAM/GPU и советует модель по силам
(«Сервис → Железо и окружение»).

### NVIDIA GPU (CUDA)

faster-whisper через CTranslate2 поддерживает CUDA 12 + cuDNN 9:

```bash
# Linux — достаточно pip-пакетов
.venv/bin/pip install -r requirements/gpu.txt
```

**Windows:** дополнительно нужны системные DLL cuBLAS/cuDNN 9. Проще всего взять
готовый архив [Purfview whisper-standalone-win (релиз `libs`)](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)
и распаковать его в `.venv\Lib\site-packages`.

После этого в настройках: Устройство → `NVIDIA GPU (CUDA)`, точность → `float16`
(или `int8_float16` при 4 ГБ VRAM), включить `Batched` — на 13-минутном аудио
ускорение до 4×.

---

## ⚡ Почему это быстро

Узкое место — железо, а не Python:

1. **Инференс вне Python.** CTranslate2 и whisper.cpp — это C++/ассемблерные
   ядра (AVX2/NEON, int8, fusion слоёв). Python только отдаёт задачи.
2. **Batched-декодинг.** `BatchedInferencePipeline` декодирует до 8 VAD-чанков
   параллельно (на GPU даёт ~4× к последовательному режиму).
3. **Silero VAD.** Тишина и шум вырезаются **до** распознавания: часовая лекция
   с паузами считается на 15–30 % быстрее.
4. **Prefetch декодирования.** Пока модель занята файлом *N*, соседний поток уже
   вытаскивает звук из файла *N+1* (включается автоматически, если хватает RAM).
5. **Модель грузится один раз** на всю очередь файлов, а не на каждый файл.
6. **Звук идёт пайпом.** `ffmpeg -f f32le … pipe:1` → numpy, без промежуточных
   WAV на диске.
7. **Декодирование видео в отдельном потоке** (`ThreadPoolExecutor`), GIL при
   этом освобождается (subprocess/C-код).
8. **Никакого PyTorch.** Только CTranslate2 + numpy: установка в разы меньше и
   быстрее, старт модели — доли секунды.

Ориентиры (Whisper L-v3 Turbo, beam 5, 1 час лекции):

| Железо | Режим | Время |
|---|---|---|
| RTX 3060 12 ГБ | float16 + batched | ~1.5–2 мин |
| RTX 3060 12 ГБ | int8, последовательно | ~6 мин |
| i7-12700K (8 ядер) | int8 | ~10–14 мин |
| i5-8250U (4 ядра) | turbo q5_0 (whisper.cpp) | ~35–45 мин |
| совсем слабый ПК | Vosk Russian | ~5 мин (но качество ниже) |

---

## 🧪 Установка движков

Базовая установка включает **faster-whisper** — его хватает для большинства
моделей. Остальные движки ставятся по требованию (программа сама предложит,
когда ты выберешь соответствующую модель):

| Движок | Пакет | Когда нужен |
|---|---|---|
| faster-whisper | в базе | CT2-модели (все Whisper/Turbo/Russian) |
| whisper.cpp | `pywhispercpp>=1.4.0` | GGML-модели, слабый CPU, квантование |
| Vosk | `vosk>=0.3.45` | Kaldi-модели, мгновенный старт |

Вручную: «Сервис → Установить движок whisper.cpp / Vosk / CUDA-библиотеки».

---

## ⌨️ Командная строка (CLI)

GUI не обязателен — всё то же самое работает из терминала (удобно для скриптов
и пакетной обработки):

```bash
# список моделей (★ — рекомендуемые)
.venv/bin/python run_app.py list-models

# диагностика: Python, venv, ffmpeg, движки, железо, модели
.venv/bin/python run_app.py doctor

# распознать один файл русской лекции
.venv/bin/python run_app.py transcribe "лекция 1.mp4" -m ru-turbo-corriolon-int8

# целую папку, в субтитры + JSON, с таймкодами слов
.venv/bin/python run_app.py transcribe ./lectures -m whisper-large-v3-turbo \
    --all-formats --words -o ./subtitles

# быстрее/точнее, конкретное устройство
.venv/bin/python run_app.py transcribe a.mp4 --device cuda --compute-type float16 \
    --batch-size 16 --beam 1

# подсказки модели (термины) и горячие слова
.venv/bin/python run_app.py transcribe a.mp4 \
    --prompt "Лекция по машинному обучению, научный стиль" \
    --hotwords "трансформер, attention, градиентный спуск, BERT"
```

<details>
<summary>Все флаги <code>transcribe</code></summary>

| Флаг | Что делает |
|---|---|
| `-m, --model` | ID или имя модели (иначе — первая установленная); при необходимости скачает |
| `-o, --output` | папка результата |
| `-f, --format` | `txt` \| `txt-time` \| `srt` \| `vtt` \| `json` \| `md` |
| `--all-formats` | сохранить сразу во все форматы |
| `-l, --language` | код языка или `auto` |
| `--beam N` | beam size (1 — быстро, 5 — баланс, 8+ — точность) |
| `--no-vad` | не вырезать тишину |
| `--no-clean` | не чистить галлюцинации/повторы |
| `--timestamps` | таймкоды в `.txt` |
| `--words` | таймкоды каждого слова (нужно для точных субтитров) |
| `--device` | `auto` \| `cpu` \| `cuda` |
| `--compute-type` | `int8` \| `int8_float16` \| `float16` \| `default` |
| `--threads N` | потоки CPU |
| `--batch-size N`, `--no-batched` | пакетный декодинг |
| `--prompt`, `--hotwords` | подсказки модели |
| `--overwrite` | перезаписывать существующие файлы |
| `--header` | добавить шапку с метаинформацией |

</details>

---

## 📁 Структура проекта

```
LocalASR/
├── main.py                     ← запускать ЭТОТ файл (проверка + установка + GUI)
├── bootstrap.py                ← подготовка .venv (только stdlib, работает на голом Python)
├── run_app.py                  ← вход для GUI/CLI уже из .venv
├── requirements/
│   ├── base.txt                ← базовые зависимости (GUI + faster-whisper)
│   ├── extras.txt              ← whisper.cpp, Vosk (ставятся по требованию)
│   ├── gpu.txt                 ← CUDA (cuBLAS + cuDNN 9)
│   └── engines.json            ← карта «движок → pip-пакеты»
├── resources/
│   ├── models_catalog.json     ← каталог моделей (размеры, языки, память, оценки)
│   └── icon.svg
├── app/
│   ├── cli.py                  ← GUI-запуск + argparse-команды
│   ├── settings.py             ← settings/settings.json (устойчив к битым файлам)
│   ├── paths.py · constants.py · logutils.py
│   ├── core/
│   │   ├── ffmpeg.py           ← поиск ffmpeg, probe, декодирование в 16 кГц моно
│   │   ├── sysinfo.py          ← CPU/RAM/GPU, рекомендация устройства и модели
│   │   ├── net.py              ← HTTP с доозагрузкой, HF API
│   │   ├── models_catalog.py   ← каталог + сканирование models/
│   │   ├── downloader.py       ← установка моделей, installed.json, автоопределение
│   │   ├── postprocess.py      ← анти-галлюцинации, абзацы, субтитры
│   │   └── exporters.py        ← txt / srt / vtt / json / md
│   ├── asr/
│   │   ├── base.py             ← общий контракт движка
│   │   ├── faster_whisper_engine.py
│   │   ├── whispercpp_engine.py
│   │   ├── vosk_engine.py
│   │   ├── registry.py         ← кэш модели, железо, потоковые экземпляры
│   │   └── transcriber.py      ← пайплайн: файл → звук → движок → текст
│   └── ui/
│       ├── main_window.py      ← главное окно
│       ├── models_panel.py     ← список моделей с фильтрами
│       ├── widgets.py          ← список файлов, карточка модели
│       ├── dialogs.py          ← прогресс, своя модель, первый запуск, настройки
│       ├── workers.py          ← QThread: установка, скачивание, распознавание
│       ├── theme.py            ← QSS-тема
│       └── compat.py           ← совместимость enum'ов PySide6 6.7–6.12
├── models/                     ← сюда ставятся модели (в git не попадает)
├── outputs/                    ← результаты по умолчанию
├── logs/                       ← localasr.log, bootstrap.log, whispercpp.log
├── scripts/                    ← dev-скрипты: doctor, e2e-проверки
└── tests/                      ← 200+ тестов (включая GUI-смоки на стабе Qt)
```

---

## 🔧 Устранение проблем

<details>
<summary><b>Мелькнуло окно консоли и пропало / «вылетает при запуске»</b></summary>

Так быть **не должно**: при любой ошибке bootstrap показывает окно с текстом
ошибки и держит консоль до нажатия Enter. Если окно всё же пропало — смотри логи:

* `logs/bootstrap.log` — что делала установка окружения;
* `logs/bootstrap_error.log` — ошибки запуска `main.py`;
* `logs/localasr.log` — ошибки самого приложения.

Запусти из терминала, чтобы увидеть всё сразу:

```bash
python main.py
```
</details>

<details>
<summary><b>«Не найден подходящий Python»</b></summary>

Нужен Python **3.10 или новее, 64-bit** (проверено вплоть до 3.14). Программа
сама найдёт подходящий интерпретатор: на Windows через лаунчер `py`, затем
`PATH` и стандартные папки установки.

```bash
py -0        # Windows: список установленных версий
python3 -V   # Linux/macOS
```

Частые причины отказа:

* установлен **32-битный** Python — скачай 64-бит («Windows installer (64-bit)»);
* Python поставлен без галочки **«Add python.exe to PATH»** и без лаунчера —
  переустанови с галочкой или запусти `main.py` из терминала полным путём;
* в системе только «заглушка» из Microsoft Store — поставь python.org-сборку.
</details>

<details>
<summary><b>Не создаётся .venv (Linux/Ubuntu)</b></summary>

```bash
sudo apt install python3-venv python3-full
```

bootstrap сам попробует план Б (`--without-pip` + `get-pip.py`), но системный
пакет надёжнее.
</details>

<details>
<summary><b>pip/uv падает или висит</b></summary>

* Проверь сеть/прокси: `python -m pip download --no-deps numpy -d /tmp/t`
* Антивирус может блокировать запись в `.venv` — добавь папку проекта в исключения.
* Полная переустановка окружения: удали папку `.venv` и запусти `main.py` заново.
* «Сервис → Проверить и починить окружение (.venv)» — доустановка без удаления.
</details>

<details>
<summary><b>Ошибка CUDA / cuDNN / cuBLAS при загрузке модели</b></summary>

Программа выдаст понятную подсказку. Варианты:

1. «Настройки → Движки и CUDA → Установить CUDA-библиотеки (pip)»;
2. Windows: распаковать DLL из
   [Purfview/whisper-standalone-win (libs)](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)
   в `.venv\Lib\site-packages`;
3. «Настройки → Скорость и железо → Устройство: CPU» — работает всегда.
</details>

<details>
<summary><b>Не хватает памяти (MemoryError)</b></summary>

* Выбери модель поменьше (`q5_0`, `small`, Vosk);
* Настройки → Точность вычислений → `int8`;
* Настройки → «Файлов одновременно» = 1;
* Закрой браузер — свободная RAM важнее общей.
</details>

<details>
<summary><b>В тексте всё равно встречается мусор</b></summary>

* Убедись, что включён чекбокс **«Анти-галлюцинации»** и **VAD**;
* Укажи язык явно (Русский) вместо «определить автоматически»;
* Добавь термины лекции в **hotwords** и опиши стиль в **initial_prompt**;
* Возьми модель поточнее (turbo → turbo Russian → large-v3).
</details>

<details>
<summary><b>В файле не нашлось аудиодорожки</b></summary>

Проверь, чем воспроизводится файл. Если звук есть, но программа его не видит —
поставь системный ffmpeg (он поддерживает больше кодеков, чем встроенный):
`winget install Gyan.FFmpeg` / `sudo apt install ffmpeg` / `brew install ffmpeg`.
Программа подхватит его автоматически (системный ffmpeg приоритетнее встроенного).
</details>

<details>
<summary><b>Qt-ошибки на Linux (xcb, libGL)</b></summary>

```bash
sudo apt install libgl1 libegl1 libxkbcommon0 libdbus-1-3 \
    libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xinerama0
# или программный рендер, если с драйвером беда:
QT_OPENGL=software python3 main.py
```
</details>

---

## 🧪 Тесты

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests -q          # 200+ тестов
```

Тесты покрывают: каталог моделей и рекомендации под железо, чистку
галлюцинаций, имена файлов и кириллицу, все форматы экспорта, разбор ссылок на
модели, bootstrap (создание venv, проверка импортов, локи, поведение без Python),
CLI, весь пайплайн целиком (на подменённом движке) и **GUI-смоки** — окно,
карточки моделей, диалоги, drag&drop, воркеры.

Если настоящий PySide6 недоступен (CI без системных библиотек Qt), тесты GUI
исполняются на стабе из `tests/stubs/` — он ловит опечатки, неверные сигнатуры
и сломанные связи сигнал/слот.

Dev-скрипты для проверки «вживую» (нужны сеть и установленные движки):

```bash
python scripts/dev_venv_check.py /tmp/venv-check    # создание venv + pip
python scripts/dev_gui_smoke.py                     # GUI на настоящем PySide6
python scripts/dev_e2e_transcribe.py vosk-ru-small-0.22
python scripts/dev_e2e_custom_model.py              # whisper.cpp + модель по ссылке
```

---

## 🤝 Вклад в проект

Приветствуются PR с новыми моделями в `resources/models_catalog.json`
(обязательные поля: `id`, `name`, `engine`, `source`, `size_bytes`, `ram_mb`,
`languages`, `quality`, `speed`, `description`).

```bash
git clone https://github.com/local-asr/LocalASR && cd LocalASR
python3 main.py                     # поднимет .venv
.venv/bin/pip install pytest ruff
.venv/bin/python -m pytest -q
```

Стиль: русский язык в интерфейсе и комментариях, type hints, без тяжёлых
зависимостей в `bootstrap.py` (только stdlib).

---

## 📄 Лицензия и благодарности

MIT — см. [LICENSE](LICENSE).

Код LocalASR распространяется свободно; **модели** имеют собственные лицензии
(Whisper — MIT, Vosk — Apache-2.0), скачиваются с Hugging Face и
alphacephei.com и остаются на твоём диске.

Спасибо проектам: [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[CTranslate2](https://github.com/OpenNMT/CTranslate2),
[whisper.cpp](https://github.com/ggml-org/whisper.cpp),
[pywhispercpp](https://github.com/absadiki/pywhispercpp),
[Vosk](https://alphacephei.com/vosk/), [Silero VAD](https://github.com/snakers4/silero-vad),
[PySide6](https://wiki.qt.io/Qt_for_Python), [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg).

<div align="center">

**Всё считается на твоём компьютере. Ничего никуда не отправляется.**

</div>
