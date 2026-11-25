@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.7.0 RAGE Stress Suite (17/17 Verify + 12M Hi-Z + 500 NPC)
echo =======================================================================
bcpack-gpu-demo.exe --verify --assets-dir "%~dp0assets"
if errorlevel 1 exit /b 1
echo.
bcpack-gpu-demo.exe --rage-stress --assets-dir "%~dp0assets"
if errorlevel 1 exit /b 1
echo.
if exist "%~dp0aaa_assets\CarConcept.glb" (
    bcpack-aaa-demo.exe --aaa-dir "%~dp0aaa_assets" --flush-bench
    if errorlevel 1 exit /b 1
)
if "%~1"=="" pause
