@echo off

set pathtracer="%~dp0/bin/raytracedao.exe"
set scenesInput="%~dp0/test_scenes.txt"

pushd bin
if NOT EXIST %pathtracer% (
    echo BatchScriptError: Pathtracer Executable does not exist. ^(at %pathtracer%^)
    popd
    EXIT /B 0
)
popd

for /f "tokens=*" %%s in ('findstr /v /c:";" %scenesInput%') do (
    REM echo %%s
    Call :render_and_denoise %%s
)

EXIT /B %ERRORLEVEL%

:render_and_denoise

pushd bin

@echo on
%pathtracer% -SCENE=%1 -TERMINATE
@echo off

popd

EXIT /B 0
