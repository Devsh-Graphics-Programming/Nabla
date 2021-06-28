include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(DENOISER_TONEMAPPER_EXAMPLE_LIBS
	${NBL_EXT_OPTIX_LIB}
	${OPTIX_DEPENDENT_LIBS}
	${NBL_EXT_MITSUBA_LOADER_LIB}
	${MITSUBA_LOADER_DEPENDENT_LIBS}
)

set(DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS
	${NBL_EXT_OPTIX_INCLUDE_DIRS}
	${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}
)

set(EXTRA_SOURCES
	CommandLineHandler.cpp
	../../src/nbl/ext/LumaMeter/CLumaMeter.cpp
	../../src/nbl/ext/ToneMapper/CToneMapper.cpp
	../../src/nbl/ext/FFT/FFT.cpp
)

nbl_create_executable_project(
	"${EXTRA_SOURCES}"
	""
	"${DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS}"
	"${DENOISER_TONEMAPPER_EXAMPLE_LIBS}"
)