@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.15.4 Clean-Room UMA Emulation Optimization Suite
echo =======================================================================
bcpack-gpu-demo.exe --uma-emulation-bench --report-json "%~dp0uma_emulation_report.json" %*
if errorlevel 1 exit /b 1
if "%~1"=="" pause
