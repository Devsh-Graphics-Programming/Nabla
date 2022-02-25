
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXT_SOURCE_FILES
	../../src/nbl/ext/DebugDraw/CDraw3DLine.cpp
	../../src/nbl/ext/DepthPyramidGenerator/DepthPyramidGenerator.cpp
)
	

nbl_create_executable_project("${EXT_SOURCE_FILES}" "" "" "")