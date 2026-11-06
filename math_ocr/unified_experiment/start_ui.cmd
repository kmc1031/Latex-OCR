@echo off
cd /d "%~dp0"
"..\.venv\Scripts\python.exe" web.py --port 8768
