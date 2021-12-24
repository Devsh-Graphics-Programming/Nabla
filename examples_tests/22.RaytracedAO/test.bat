@echo off

set pathtracer="raytracedao.exe"

pushd bin
if NOT EXIST %pathtracer% (
    echo BatchScriptError: Pathtracer Executable does not exist. ^(at %pathtracer%^)
    popd
    EXIT /B 0
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

popd

EXIT /B 0
