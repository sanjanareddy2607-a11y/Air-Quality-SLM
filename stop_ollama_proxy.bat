@echo off
rem ============================================================================
rem  Air Quality SLM integration - stop script
rem ============================================================================
rem  Stops the watchdog first, then Ollama Desktop, then the Python proxy and
rem  the backend "ollama serve". Only processes whose command line belongs to
rem  this integration are terminated; unrelated python.exe programs are never
rem  touched. Every stopped process is reported by PID.
rem
rem  Options:
rem    stop_ollama_proxy.bat              stop everything
rem    stop_ollama_proxy.bat /keepdesktop stop services, leave Desktop open
rem ============================================================================
setlocal
set "PY=python"
where python >nul 2>&1 || set "PY=C:\Python314\python.exe"
set "KEEP="
if /I "%~1"=="/keepdesktop" set "KEEP=--keep-desktop"

"%PY%" "%~dp0scripts\integration_ctl.py" stop %KEEP%
set "RC=%ERRORLEVEL%"

echo.
pause
exit /b %RC%
