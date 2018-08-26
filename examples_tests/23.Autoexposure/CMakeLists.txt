
set(EXTRA_SOURCES ../../ext/AutoExposure/CToneMapper.cpp)
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(SEND_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()