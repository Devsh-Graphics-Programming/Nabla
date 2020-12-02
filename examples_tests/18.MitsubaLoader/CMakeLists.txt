
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(MITSUBA_EXAMPLE_LIBS
	${NBL_EXT_MITSUBA_LOADER_LIB}
	${MITSUBA_LOADER_DEPENDENT_LIBS}
)

nbl_create_executable_project(
	""
	""
	"${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}"
	"${MITSUBA_EXAMPLE_LIBS}"
)