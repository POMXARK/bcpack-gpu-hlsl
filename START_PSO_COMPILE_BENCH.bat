@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.15.4 Clean-Room Real-Time PSO Compilation Suite
echo =======================================================================
bcpack-gpu-demo.exe --pso-compile-bench --report-json "%~dp0pso_compile_report.json" %*
if errorlevel 1 exit /b 1
if "%~1"=="" pause
