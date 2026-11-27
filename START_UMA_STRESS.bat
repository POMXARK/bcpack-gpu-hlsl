@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.15.4 Clean-Room UMA Stress Showcase ([U] + [1]..[4])
echo =======================================================================
target\release\bcpack-aaa-demo.exe --uma-stress --no-vsync --show-hud %*
if "%~1"=="" pause
