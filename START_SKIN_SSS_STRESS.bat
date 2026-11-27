@echo off
setlocal
cd /d "%~dp0"
if not exist "media" mkdir "media"

set "BIN=target\release\bcpack-gpu-demo.exe"
if not exist "%BIN%" (
    echo [BUILD] Building release binary...
    cargo build --release --workspace
    if errorlevel 1 exit /b 1
)

echo [RUN] Executing --skin-sss-stress --debug ...
"%BIN%" --skin-sss-stress --debug --report-json "%~dp0skin_sss_report.json" --screenshot-dir "%~dp0media" %*
if errorlevel 1 exit /b 1

echo [DONE] Report: %~dp0skin_sss_report.json
echo [DONE] Screenshots: %~dp0media\skin_sss_naive.png and %~dp0media\skin_sss_opt.png
