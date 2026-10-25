@echo off
rem Creates the "LocalASR" shortcut on the Desktop with the app icon.
cd /d "%~dp0"
wscript.exe "%~dp0tools\make_shortcut.vbs"
