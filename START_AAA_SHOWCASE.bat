@echo off
cd /d "%~dp0"
echo =======================================================================
echo   BCPack GPU AAA Showcase v0.7.0 (2K PBR + Sunset HDRI + DXR 1.1)
echo   [T] Flush VRAM Wave  [V] X-Ray Heatmap  [SPACE] GPU/CPU  [R] DXR
echo   [G] RTGI Denoiser    [C] Camera Mode    [WASD/Arrows] Move (FreeCam)
echo =======================================================================
bcpack-aaa-demo.exe --aaa-dir "%~dp0aaa_assets" %*
