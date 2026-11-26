@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU v0.15.1 Open-World Pursuit Stress (6-Fmt Zero-Copy + Pager)
echo =======================================================================
target\release\bcpack-aaa-demo.exe --pursuit-stress --no-vsync --show-hud %*
if "%~1"=="" pause
