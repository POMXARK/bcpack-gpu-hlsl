@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU AAA Showcase v0.8.0 (4K BC7/BC5 + 12.41M City + 500 NPCs + DXR 1.1)
echo   [H] HUD Sliders (FPS Cap + Sim Speed)  [/] FPS  [-/=] Sim  [P] Pause
echo   [T] Flush VRAM Wave  [V] X-Ray Heatmap  [F] SFS Budget  [L] VSync (Monitor Hz Cap)
echo   [SPACE] GPU/CPU  [R] DXR  [G] RTGI  [C] Camera Mode  [ESC] Quit
echo =======================================================================
bcpack-aaa-demo.exe --aaa-dir "%~dp0aaa_assets" %*
