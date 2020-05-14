
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
	../../ext/LumaMeter/CGLSLLumaBuiltinIncludeLoader.cpp
	../../ext/LumaMeter/CLumaMeter.cpp
	../../ext/ToneMapper/CToneMapper.cpp
)

irr_create_executable_project("${EXAMPLE_SOURCES}" "" "" "")