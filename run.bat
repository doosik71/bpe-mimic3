@echo off
rem run.bat -- dispatch to a project Python script under scripts\.
rem
rem   run <script-name> [args...]   =^>  uv run python scripts\<script-name>.py [args...]
rem
rem The trailing ".py" is optional, so "run eval-model" and "run eval-model.py"
rem are equivalent. With no arguments, the available scripts are listed.
rem
rem Examples:
rem   run eval-calib-model --help
rem   run train-wcrc-model --dataset-dir data\dataset-v2 --models-dir data\models-v2
rem   run eval-continuous-model data\models\mtae_wcrc
setlocal EnableExtensions
cd /d "%~dp0"

if "%~1"=="" (
    call :list
    exit /b 0
)

set "NAME=%~1"
if /I "%NAME:~-3%"==".py" set "NAME=%NAME:~0,-3%"
set "TARGET=scripts\%NAME%.py"

if not exist "%TARGET%" (
    echo run: unknown script '%NAME%' ^(no %TARGET%^) 1>&2
    echo. 1>&2
    call :list 1>&2
    exit /b 1
)

rem Pass every argument after the script name through unchanged.
set "REST="
for /f "tokens=1,* delims= " %%a in ("%*") do set "REST=%%b"

uv run python "%TARGET%" %REST%
exit /b %ERRORLEVEL%

:list
echo Usage: run ^<script-name^> [args...]
echo        run ^<script-name^> --help
echo.
echo Available scripts:
for %%f in (scripts\*.py) do echo   %%~nf
exit /b 0
