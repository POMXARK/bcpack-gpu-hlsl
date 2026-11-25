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
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bcpack_bc7_rans.dxil" "%~dp0bcpack_bc7_rans.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0gfx10_detile.dxil" "%~dp0gfx10_detile.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0rt_verify.dxil" "%~dp0rt_verify.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0scene_vs.dxil" "%~dp0scene.hlsl" || exit /b 1
"%DXC%" -T ps_6_5 -E PSMain -HV 2021 -O3 -Fo "%~dp0scene_ps.dxil" "%~dp0scene.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0sky_vs.dxil" "%~dp0sky.hlsl" || exit /b 1
"%DXC%" -T ps_6_0 -E PSMain -O3 -Fo "%~dp0sky_ps.dxil" "%~dp0sky.hlsl" || exit /b 1
"%DXC%" -T as_6_5 -E ASMain -HV 2021 -O3 -Fo "%~dp0meshlet_as.dxil" "%~dp0meshlet_as.hlsl" || exit /b 1
"%DXC%" -T ms_6_5 -E MSMain -HV 2021 -O3 -Fo "%~dp0meshlet_ms.dxil" "%~dp0meshlet_ms.hlsl" || exit /b 1
"%DXC%" -T ps_6_5 -E PSMain -HV 2021 -O3 -Fo "%~dp0meshlet_ps.dxil" "%~dp0meshlet_ps.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0rtgi_denoise.dxil" "%~dp0rtgi_denoise.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0rtgi_comp_vs.dxil" "%~dp0rtgi_composite.hlsl" || exit /b 1
"%DXC%" -T ps_6_0 -E PSMain -O3 -Fo "%~dp0rtgi_comp_ps.dxil" "%~dp0rtgi_composite.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0aaa_sky_vs.dxil" "%~dp0aaa_sky.hlsl" || exit /b 1
"%DXC%" -T ps_6_0 -E PSMain -O3 -Fo "%~dp0aaa_sky_ps.dxil" "%~dp0aaa_sky.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0aaa_pbr_vs.dxil" "%~dp0aaa_pbr.hlsl" || exit /b 1
"%DXC%" -T ps_6_6 -E PSMain -HV 2021 -O3 -Fo "%~dp0aaa_pbr_ps.dxil" "%~dp0aaa_pbr.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0aaa_bloom.dxil" "%~dp0aaa_bloom.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0aaa_taa.dxil" "%~dp0aaa_taa.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0aaa_autoexp.dxil" "%~dp0aaa_autoexp.hlsl" || exit /b 1
"%DXC%" -T vs_6_0 -E VSMain -O3 -Fo "%~dp0aaa_comp_vs.dxil" "%~dp0aaa_composite.hlsl" || exit /b 1
"%DXC%" -T ps_6_0 -E PSMain -O3 -Fo "%~dp0aaa_comp_ps.dxil" "%~dp0aaa_composite.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0bindless_paging.dxil" "%~dp0bindless_paging.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0hiz_build.dxil" "%~dp0hiz_build.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0hiz_cull.dxil" "%~dp0hiz_cull.hlsl" || exit /b 1
"%DXC%" -T cs_6_6 -E CSMain -HV 2021 -O3 -Fo "%~dp0crowd_skin.dxil" "%~dp0crowd_skin.hlsl" || exit /b 1
echo Shaders compiled successfully.
