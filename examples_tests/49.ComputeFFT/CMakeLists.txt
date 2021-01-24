
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXAMPLE_SOURCES
	../../src/nbl/ext/LumaMeter/CLumaMeter.cpp
	../../src/nbl/ext/ToneMapper/CToneMapper.cpp
	../../src/nbl/ext/FFT/FFT.cpp
)

nbl_create_executable_project("${EXAMPLE_SOURCES}" "" "" "")