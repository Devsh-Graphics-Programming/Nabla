
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

irr_create_executable_project(
	../../ext/DebugDraw/CDraw3DLine.cpp
	"" 
	"${IRR_EXT_BULLET_INCLUDE_DIRS}" 
	"${IRR_EXT_BULLET_LIB}"
)
