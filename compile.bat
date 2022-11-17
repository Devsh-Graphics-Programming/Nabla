
@echo off 

setlocal enabledelayedexpansion


:: List of shaders to compile (relative to include/nbl/builtin/hlsl/)
set file_path[0]=common
set file_path[1]=algorithm
set file_path[2]=ieee754
set file_path[3]=limits/numeric
set file_path[4]=math/complex
set file_path[5]=math/constants



set "XOUTPUT_PATH=include/nbl/builtin/hlsl/xoutput/"
set "HLSL_PATH=include/nbl/builtin/hlsl/"


:: Count elements in "file_path" array
set /a len=0 
:Loop
if defined file_path[%len%] ( 
	set /a len+=1
	GOTO :Loop 
)
set /a len-=1


cd include/nbl/builtin/hlsl


:: Create non-existing file paths
for /L %%a in (0, 1, %len%) do (
	if not exist "xoutput\!file_path[%%a]!" (
		mkdir "xoutput/!file_path[%%a]!"
		rmdir "xoutput\!file_path[%%a]!"
	)
)

cd ../../../../

echo:
echo:
echo  Compiling HLSL shaders...
echo:
echo:
:: Compile all
for /L %%a in (0, 1, %len%) do (
	3rdparty\dxc\dxc\bin\x64\dxc.exe -HV 2021 -T  lib_6_7 -I include/ -Zi -Qembed_debug -Fo %XOUTPUT_PATH%!file_path[%%a]!  %HLSL_PATH%!file_path[%%a]!.hlsl
)
echo:
echo:
echo    Done Compiling!
echo  Compiled shaders are in - "%XOUTPUT_PATH%"
echo:
echo:

pause