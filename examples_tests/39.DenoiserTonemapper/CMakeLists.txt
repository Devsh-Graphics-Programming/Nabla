include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(DENOISER_TONEMAPPER_EXAMPLE_LIBS
	${IRR_EXT_OPTIX_LIB}
	${OPTIX_DEPENDENT_LIBS}
)

set(DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS
	${IRR_EXT_OPTIX_INCLUDE_DIRS}
)

set(EXTRA_SOURCES
	CommandLineHandler.cpp
	../../ext/LumaMeter/CGLSLLumaBuiltinIncludeLoader.cpp
	../../ext/LumaMeter/CLumaMeter.cpp
	../../ext/ToneMapper/CToneMapper.cpp
)

irr_create_executable_project(
	"${EXTRA_SOURCES}"
	""
	"${DENOISER_TONEMAPPER_EXAMPLE_INCLUDE_DIRS}"
	"${DENOISER_TONEMAPPER_EXAMPLE_LIBS}"
)