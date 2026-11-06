@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0ui_server.py" --open %*
if errorlevel 1 pause
