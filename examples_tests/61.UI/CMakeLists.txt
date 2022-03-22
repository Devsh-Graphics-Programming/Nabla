include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

list(
	APPEND 
	EXTRA_SOURCES 
	"UI_System.cpp" 
	"UI_System.hpp"
)

nbl_create_executable_project("${EXTRA_SOURCES}" "" "" "")