
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(OPTIX_PATH_TRACER_EXAMPLE_LIBS
	${IRR_EXT_OPTIX_LIB}
	${OPTIX_DEPENDENT_LIBS}
)

set(OPTIX_PATH_TRACER_EXAMPLE_INCLUDE_DIRS
	${IRR_EXT_OPTIX_INCLUDE_DIRS}
)

irr_create_executable_project(
	""
	""
	"${OPTIX_PATH_TRACER_EXAMPLE_INCLUDE_DIRS}"
	"${OPTIX_PATH_TRACER_EXAMPLE_LIBS}"
)