@echo off
rem =====================================================================
rem  LocalASR - запуск для тех, кому удобнее батник, чем двойной клик
rem  по main.py. Поведение то же самое: проверка окружения, установка
rem  недостающего в .venv и запуск GUI без консоли.
rem =====================================================================
chcp 65001 >nul
cd /d "%~dp0"
set "LOCALASR_NO_PAUSE="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 main.py %*
) else (
    python main.py %*
)
if errorlevel 1 pause
