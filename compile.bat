
@echo off 

setlocal enabledelayedexpansion


:: List of shaders to compile (paths relative to include/nbl/builtin/hlsl/)
set file_path[0]=common
set file_path[1]=algorithm
set file_path[2]=ieee754

set file_path[3]=limits/numeric

set file_path[4]=math/complex
set file_path[5]=math/constants
set file_path[6]=math/functions
set file_path[7]=math/binary_operator_functions
set file_path[8]=math/quaternions

set file_path[9]=scene/animation
set file_path[10]=scene/keyframe
set file_path[11]=scene/node

set file_path[12]=format/constants
set file_path[13]=format/decode
set file_path[14]=format/encode

set file_path[15]=colorspace/decodeCIEXYZ
set file_path[16]=colorspace/encodeCIEXYZ
set file_path[17]=colorspace/EOTF
set file_path[18]=colorspace/OETF

set file_path[19]=shapes/aabb
set file_path[20]=shapes/rectangle
set file_path[21]=shapes/triangle
set file_path[22]=shapes/frustum

set file_path[23]=random/xoroshiro

set file_path[24]=utils/compressed_normal_matrix_t
set file_path[25]=utils/acceleration_structures
set file_path[26]=utils/common
set file_path[27]=utils/culling


rem set file_path[23]=bxdf/fresnel






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
	3rdparty\dxc\dxc\bin\x64\dxc.exe -HV 2021 -T  lib_6_7 -I include/ -Zi -Qembed_debug -Fo %XOUTPUT_PATH%!file_path[%%a]!.bin  %HLSL_PATH%!file_path[%%a]!.hlsl
)
echo:
echo:
echo    Done Compiling!
echo  Compiled shaders are in - "%XOUTPUT_PATH%"
echo:
echo:

pause