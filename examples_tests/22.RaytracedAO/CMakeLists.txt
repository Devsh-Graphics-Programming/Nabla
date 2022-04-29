
include(common RESULT_VARIABLE RES)
if(NOT RES)
	message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(RAY_TRACED_AO_EXAMPLE_LIBS
	${NBL_EXT_RADEON_RAYS_LIB}
	${RADEON_RAYS_DEPENDENT_LIBS}
	${NBL_EXT_MITSUBA_LOADER_LIB}
	${MITSUBA_LOADER_DEPENDENT_LIBS}
)

set(RAY_TRACED_AO_EXAMPLE_INCLUDE_DIRS
	${NBL_EXT_RADEON_RAYS_INCLUDE_DIRS}
	${NBL_EXT_MITSUBA_LOADER_INCLUDE_DIRS}
)

if(NBL_BUILD_OPTIX)
	set(RAY_TRACED_AO_EXAMPLE_LIBS
		${RAY_TRACED_AO_EXAMPLE_LIBS}
		${NBL_EXT_OPTIX_LIB}
		${OPTIX_DEPENDENT_LIBS}
	)

	set(RAY_TRACED_AO_EXAMPLE_INCLUDE_DIRS
		${RAY_TRACED_AO_EXAMPLE_INCLUDE_DIRS}
		${NBL_EXT_OPTIX_INCLUDE_DIRS}
	)
endif()

set(EXTRA_SOURCES
	../../src/nbl/ext/DebugDraw/CDraw3DLine.cpp
	../../src/nbl/ext/EnvmapImportanceSampling/EnvmapImportanceSampling.cpp
	Renderer.cpp
	CommandLineHandler.cpp
)

nbl_create_executable_project(
	"${EXTRA_SOURCES}"
	""
	"${RAY_TRACED_AO_EXAMPLE_INCLUDE_DIRS}"
	"${RAY_TRACED_AO_EXAMPLE_LIBS}"
)