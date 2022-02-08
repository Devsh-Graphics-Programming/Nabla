@echo off

set denoiser_dir="%~dp0../39.DenoiserTonemapper/bin"
if NOT EXIST %denoiser_dir%/denoisertonemapper.exe (
    echo BatchScriptError: Denoiser Executable does not exist. ^(at %denoiser%^)
    REM Don't Exit when denoiser is not found
    REM popd
    REM EXIT /B 0
)

REM 1.ColorFile 2.AlbedoFile 3.NormalFile 4.BloomPsfFilePath(STRING) 5.BloomScale(FLOAT) 6.BloomIntensity(FLOAT) 7.TonemapperArgs(STRING)
call :denoise %1 %2 %3 %4 %5 %6 %7


EXIT /B %ERRORLEVEL%

:denoise

set color_file="%~dpn1.exr"
set albedo_file="%~dpn2.exr"
set normal_file="%~dpn3.exr"
set output_file="%~dpn1_denoised.exr"

if NOT EXIST %color_file% (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %color_file%^)
    EXIT /B 0
)
if NOT EXIST %albedo_file% (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %albedo_file%^)
    EXIT /B 0
)
if NOT EXIST %normal_file% (
    echo BatchScriptError: Denoiser input file doesn't exist. ^(at %normal_file%^)
    EXIT /B 0
)
set bloom_file=%~f4
set bloom_scale=%~5
set bloom_intensity=%~6
set tonemapper_args=%~7
@echo on
pushd %denoiser_dir%
denoisertonemapper.exe -COLOR_FILE=%color_file% -ALBEDO_FILE=%albedo_file% -NORMAL_FILE=%normal_file% -OUTPUT=%output_file% -DENOISER_EXPOSURE_BIAS=0.0 -DENOISER_BLEND_FACTOR=0.0 -BLOOM_PSF_FILE=%bloom_file% -BLOOM_RELATIVE_SCALE=%bloom_scale% -BLOOM_INTENSITY=%bloom_intensity% -TONEMAPPER=%tonemapper_args% -CAMERA_TRANSFORM=1,0,0,0,1,0,0,0,1
popd
@echo off

EXIT /B 0
