@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.8.0 Cinematic AAA Benchmark (60s 3-Act Automated Run)
echo   4K BC7/BC5 + Active SFS + 12.41M Mesh-Shader City + 500 NPCs + DXR 1.1
echo   Output: Avg FPS / 1%% Low / 0.1%% Low / SFS VRAM Saved / Hi-Z Culled
echo =======================================================================
bcpack-aaa-demo.exe --aaa-dir "%~dp0aaa_assets" --benchmark-run --benchmark-secs 60 %*
if errorlevel 1 exit /b 1
if "%~1"=="" pause
