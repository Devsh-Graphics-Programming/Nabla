@echo off

set pathtracer="raytracedao.exe"
set denoiser="../../39.DenoiserTonemapper/bin/denoisertonemapper.exe"

pushd bin
if NOT EXIST %pathtracer% (
    echo BatchScriptError: Pathtracer Executable does not exist. ^(at %pathtracer%^)
    popd
    EXIT /B 0
)

if NOT EXIST %denoiser% (
    echo BatchScriptError: Denoiser Executable does not exist. ^(at %denoiser%^)
    REM Don't Exit when denoiser is not found
    REM popd
    REM EXIT /B 0
)
popd

for /f "tokens=*" %%s in ('findstr /v /c:";" test_scenes.txt') do (
    REM echo %%s
    Call :render_and_denoise %%s
)

EXIT /B %ERRORLEVEL%

:render_and_denoise

pushd bin

@echo on
%pathtracer% -SCENE=%1 -TERMINATE
@echo off

REM throw the first parameter away and iterate through the second to last param to denoise
shift
set params=%1
:loop
Call :denoise %1
shift
if [%1]==[] goto afterloop
set params=%params% %1
goto loop
:afterloop

popd

EXIT /B 0

:denoise
if NOT EXIST %~dpn1.exr (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %~dpn1.exr^)
    EXIT /B 0
)
@echo on
%denoiser% -COLOR_FILE=%~dpn1.exr -OUTPUT=%~dpn1_DENOISED.exr -DENOISER_EXPOSURE_BIAS=0.0 -DENOISER_BLEND_FACTOR=0.0 -BLOOM_PSF_FILE=../../media/kernels/physical_flare_512.exr -BLOOM_RELATIVE_SCALE=0.01 -BLOOM_INTENSITY=0.0 -TONEMAPPER=ACES=0.4,0.8 -CAMERA_TRANSFORM=1,0,0,0,1,0,0,0,1
@echo off

EXIT /B 0
