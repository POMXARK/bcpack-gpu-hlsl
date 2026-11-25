@echo off
setlocal
set "DXC="
for /f "delims=" %%F in ('where dxc.exe 2^>nul') do (
    set "DXC=%%F"
    goto :found
)
for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files (x86)\Windows Kits\10\bin\10.*" 2^>nul') do (
    if exist "C:\Program Files (x86)\Windows Kits\10\bin\%%D\x64\dxc.exe" (
        set "DXC=C:\Program Files (x86)\Windows Kits\10\bin\%%D\x64\dxc.exe"
        goto :found
    )
)
echo ERROR: dxc.exe not found in PATH or Windows 10 SDK.
exit /b 1

:found
echo Using DXC: "%DXC%"
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bcpack_decode.dxil" "%~dp0bcpack_decode.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0scene_vs.dxil" "%~dp0scene.hlsl" || exit /b 1
"%DXC%" -T ps_6_0 -E PSMain -O3 -Fo "%~dp0scene_ps.dxil" "%~dp0scene.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0sky_vs.dxil" "%~dp0sky.hlsl" || exit /b 1
"%DXC%" -T ps_6_0 -E PSMain -O3 -Fo "%~dp0sky_ps.dxil" "%~dp0sky.hlsl" || exit /b 1
echo Shaders compiled successfully.
