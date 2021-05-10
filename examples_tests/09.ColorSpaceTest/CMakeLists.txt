include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(EXTRA_SOURCES
	ApplicationHandler.cpp
)

nbl_create_executable_project(
	"${EXTRA_SOURCES}"
	""
	"${DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS}"
	""
)