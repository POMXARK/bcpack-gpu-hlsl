@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.15.0 Open-World DX12 Bottleneck Suite
echo =======================================================================
bcpack-gpu-demo.exe --bottleneck-suite --assets-dir "%~dp0assets" --report-json "%~dp0bottleneck_report.json"
if errorlevel 1 exit /b 1
if "%~1"=="" pause
