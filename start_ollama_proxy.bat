@echo off
rem ============================================================================
rem  Air Quality SLM - Ollama Desktop integration launcher
rem ============================================================================
rem  Topology:
rem    Ollama Desktop client  ->  127.0.0.1:11434  (Python proxy)
rem    Python proxy           ->  127.0.0.1:11435  (real ollama serve)
rem
rem  The work is done by scripts\integration_ctl.py, which:
rem    1. stops only this integration's processes (matched by command line)
rem    2. makes sure ports 11434 / 11435 are free
rem    3. starts ollama serve on 11435 and waits until it answers
rem    4. starts exactly one proxy on 11434 and waits until it answers
rem    5. verifies each port belongs to the process it just started
rem    6. verifies proxy -> backend forwarding
rem    7. checks for duplicate processes and starts the watchdog
rem    8. launches Ollama Desktop and waits for it to connect to 11434
rem
rem  Usage:
rem    start_ollama_proxy.bat            interactive
rem    start_ollama_proxy.bat /quiet     no pause (for Task Scheduler)
rem ============================================================================
setlocal
set "PY=python"
where python >nul 2>&1 || set "PY=C:\Python314\python.exe"

"%PY%" "%~dp0scripts\integration_ctl.py" start
set "RC=%ERRORLEVEL%"

if /I "%~1"=="/quiet" exit /b %RC%
if not "%RC%"=="0" (
    echo.
    echo Startup reported problems ^(exit code %RC%^).
    echo Run check_ollama_proxy.bat for a detailed diagnosis.
)
echo.
pause
exit /b %RC%
