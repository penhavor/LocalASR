@echo off
rem =====================================================================
rem  LocalASR launcher. Uses the portable Python in runtime\python; on a
rem  fresh copy that folder is downloaded automatically on the first run
rem  (tools\download_runtime.ps1, ~15 MB, needs internet). If the runtime
rem  cannot be obtained, falls back to the system Python.
rem =====================================================================
chcp 65001 >nul
cd /d "%~dp0"
set "LOCALASR_NO_PAUSE="

if not exist "runtime\python\python.exe" call :get_runtime

if exist "runtime\python\python.exe" (
    if exist "runtime\python\.localasr-env.json" (
        if "%~1"=="" (
            rem environment ready: start the GUI without a console
            start "" "runtime\python\pythonw.exe" launcher\run_app.py
            exit /b 0
        )
        rem headless CLI mode keeps the console for output
        "runtime\python\python.exe" launcher\run_app.py %*
        exit /b %errorlevel%
    )
    rem first run or repair: visible console with the install progress
    "runtime\python\python.exe" launcher\main.py %*
    if errorlevel 1 pause
    exit /b 0
)

rem no portable runtime - use the system Python
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 launcher\main.py %*
) else (
    python launcher\main.py %*
)
if errorlevel 1 pause
exit /b 0

:get_runtime
rem download the portable Python via PowerShell (built into Windows)
where powershell >nul 2>nul
if errorlevel 1 (
    echo LocalASR: PowerShell not found - cannot download the portable Python.
    exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\download_runtime.ps1"
exit /b %errorlevel%
