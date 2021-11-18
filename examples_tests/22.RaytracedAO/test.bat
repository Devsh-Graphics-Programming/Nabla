@echo off

set pathtracer="raytracedao_rwdi.exe"
set denoiser="../../39.DenoiserTonemapper/bin/denoisertonemapper_rwdi.exe"

pushd bin
if NOT EXIST %pathtracer% (
    echo "Pathtracer Executable (at %pathtracer%) doesn't exist."
)

if NOT EXIST %denoiser% (
    echo "Denoiser Executable  (at %denoiser%) doesn't exist."
)
popd

for /f "tokens=*" %%s in ('findstr /v /c:";" test_scenes.txt') do (
    pushd bin
    Call :render_and_denoise %%s
    popd
    REM echo %%s
)

EXIT /B %ERRORLEVEL%
:render_and_denoise

@echo on
%pathtracer% -SCENE=%~1 -TERMINATE
@echo off

if NOT EXIST %~dpn2.exr (
    echo "Denoiser input file doesn't exist. (at %~dpn2.exr) "
    EXIT /B 0
)

@echo on
%denoiser% -COLOR_FILE=%~dpn2.exr -OUTPUT=%~dpn2_DENOISED.exr -DENOISER_EXPOSURE_BIAS=0.0 -DENOISER_BLEND_FACTOR=0.0 -BLOOM_PSF_FILE=../../media/kernels/physical_flare_512.exr -BLOOM_RELATIVE_SCALE=0.01 -BLOOM_INTENSITY=0.0 -TONEMAPPER=ACES=0.4,0.8 -CAMERA_TRANSFORM=1,0,0,0,1,0,0,0,1
@echo off

EXIT /B 0
