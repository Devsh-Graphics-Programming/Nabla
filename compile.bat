
setlocal enabledelayedexpansion

set file_path[0]=common
set file_path[1]=algorithm
set file_path[2]=ieee754
set file_path[3]=limits/numeric


set len=0 

:: To iterate the element of array
:Loop 

:: It will check if the element is defined or not
if defined file_path[%len%] ( 
set /a len+=1
GOTO :Loop 
)

set "XOUTPUT_PATH=include/nbl/builtin/hlsl/xoutput/"

set "HLSL_PATH=include/nbl/builtin/hlsl/"


cd include/nbl/builtin/hlsl


(for /L %%a in (0, 1, %len%) do (
	if not exist "xoutput\!file_path[%%a]!" mkdir "xoutput/!file_path[%%a]!"
	rmdir "xoutput\!file_path[%%a]!"
))

cd ../../../../

(for /L %%a in (0, 1, %len%) do (
	3rdparty\dxc\dxc\bin\x64\dxc.exe -HV 2021 -T  lib_6_7 -I include/ -Zi -Fo %XOUTPUT_PATH%!file_path[%%a]!  %HLSL_PATH%!file_path[%%a]!.hlsl
))

pause