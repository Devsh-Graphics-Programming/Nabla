include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXTRA_SOURCES
	CommandLineHandler.cpp
)

irr_create_executable_project("${EXTRA_SOURCES}" "" "" "")