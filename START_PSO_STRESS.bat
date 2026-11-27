@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.15.4 Clean-Room Real-Time PSO Stress ([O] + [1]..[4])
echo =======================================================================
target\release\bcpack-aaa-demo.exe --pso-stress --no-vsync --show-hud %*
if "%~1"=="" pause
