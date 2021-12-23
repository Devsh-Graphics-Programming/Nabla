@echo off

set denoiser_dir="%~dp0../39.DenoiserTonemapper/bin"
if NOT EXIST %denoiser_dir%/denoisertonemapper.exe (
    echo BatchScriptError: Denoiser Executable does not exist. ^(at %denoiser%^)
    REM Don't Exit when denoiser is not found
    REM popd
    REM EXIT /B 0
)

call :denoise %1 %2 %3

EXIT /B %ERRORLEVEL%

:denoise
if NOT EXIST %~dpn1.exr (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %~dpn1.exr^)
    EXIT /B 0
)
if NOT EXIST %~dpn2.exr (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %~dpn2.exr^)
    EXIT /B 0
)
if NOT EXIST %~dpn3.exr (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %~dpn3.exr^)
    EXIT /B 0
)
set color_file=%~dpn1.exr
set albedo_file=%~dpn2.exr
set normal_file=%~dpn3.exr
set output_file=%~dpn1_denoised.exr
@echo on
pushd %denoiser_dir%
denoisertonemapper.exe -COLOR_FILE=%color_file% -ALBEDO_FILE=%albedo_file% -NORMAL_FILE=%normal_file% -OUTPUT=%output_file% -DENOISER_EXPOSURE_BIAS=0.0 -DENOISER_BLEND_FACTOR=0.0 -BLOOM_PSF_FILE=../../media/kernels/physical_flare_512.exr -BLOOM_RELATIVE_SCALE=0.1 -BLOOM_INTENSITY=0.1 -TONEMAPPER=ACES=0.4,0.8 -CAMERA_TRANSFORM=1,0,0,0,1,0,0,0,1
popd
@echo off

EXIT /B 0
