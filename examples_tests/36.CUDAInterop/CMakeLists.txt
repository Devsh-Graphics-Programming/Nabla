
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(CUDA_INTEROP_EXAMPLE_INCLUDE_DIRS
	${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

irr_create_executable_project(
	""
	""
	"${CUDA_INTEROP_EXAMPLE_INCLUDE_DIRS}"
	""
)