@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU AAA Showcase v0.7.0 (2K PBR + Sunset HDRI + DXR 1.1)
echo   [T] Flush VRAM Wave  [V] X-Ray Heatmap  [L] VSync (Monitor Hz Cap)
echo   [SPACE] GPU/CPU  [R] DXR  [G] RTGI  [C] Camera Mode  [ESC] Quit
echo =======================================================================
bcpack-aaa-demo.exe --aaa-dir "%~dp0aaa_assets" %*
