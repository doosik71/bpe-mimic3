@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
uv run python "%SCRIPT_DIR%..\scripts\collect-result.py" %*
