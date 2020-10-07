
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
	../../src/irr/ext/LumaMeter/CLumaMeter.cpp
	../../src/irr/ext/ToneMapper/CToneMapper.cpp
)

irr_create_executable_project("${EXAMPLE_SOURCES}" "" "" "")