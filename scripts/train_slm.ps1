# Air Quality SLM Training Script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   AIR QUALITY SLM: LAUNCHING AUTONOMOUS TRAINING        " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

if (Test-Path $VenvPython) {
    Write-Host "Using virtual environment: $VenvPython" -ForegroundColor Yellow
    & $VenvPython (Join-Path $ProjectRoot "src\train_slm.py") @args
} else {
    Write-Host "Using default python on PATH" -ForegroundColor Yellow
    python (Join-Path $ProjectRoot "src\train_slm.py") @args
}
