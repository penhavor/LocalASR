# LocalASR — команды разработчика
PYTHON ?= python3
VENV   := .venv
BIN    := $(VENV)/bin
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts
endif

.PHONY: help run install test lint gui-smoke doctor clean

help:            ## показать справку
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

run:               ## подготовить окружение и запустить GUI
	$(PYTHON) main.py

install:           ## создать/починить .venv и поставить базовые зависимости
	$(PYTHON) main.py --force-install
	$(BIN)/python -m pip install -r requirements/base.txt

test:              ## запустить тесты
	$(BIN)/python -m pytest tests -q

lint:              ## ruff (если установлен)
	$(BIN)/python -m ruff check app bootstrap.py main.py run_app.py || true

gui-smoke:         ## smoke-тест GUI на настоящем PySide6
	QT_QPA_PLATFORM=offscreen $(BIN)/python scripts/dev_gui_smoke.py

doctor:            ## диагностика окружения (без GUI)
	$(PYTHON) main.py --doctor
	-$(BIN)/python run_app.py doctor

clean:             ## удалить кэш и результаты (модели и .venv не трогает)
	rm -rf outputs/* logs/* .cache **/__pycache__ .pytest_cache
