@echo off
rem ============================================================================
rem  Air Quality SLM integration - diagnostic health check
rem ============================================================================
rem  Read-only. Checks:
rem    1. backend process (ollama serve)
rem    2. proxy process (ollama_proxy.py)
rem    3. watchdog process (proxy_watchdog.py)
rem    4. owner of port 11434 (must be the proxy, exactly one listener)
rem    5. owner of port 11435 (must be ollama, exactly one listener)
rem    6. backend HTTP  /api/version
rem    7. proxy   HTTP  /api/version  (end-to-end through the proxy)
rem    8. model list through the proxy (/api/tags)
rem    9. latest proxy_access.log entry
rem   10. Ollama Desktop connection to 11434
rem   11. duplicate process check
rem
rem  Exit code: 0 = no failures, 1 = at least one FAIL.
rem ============================================================================
setlocal
set "PY=python"
where python >nul 2>&1 || set "PY=C:\Python314\python.exe"

"%PY%" "%~dp0scripts\integration_ctl.py" status
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo Emergency recovery:
    echo   1. stop_ollama_proxy.bat
    echo   2. start_ollama_proxy.bat
    echo   3. In Ollama Desktop select "air-quality-slm-improved:latest" and send "Hey"
    echo   4. Run check_ollama_proxy.bat again and confirm item 9 shows that request
    echo.
)
pause
exit /b %RC%
