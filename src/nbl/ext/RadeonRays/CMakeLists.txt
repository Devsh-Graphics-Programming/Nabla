include(${NBL_ROOT_PATH}/cmake/common.cmake)
set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include/nbl/ext/RadeonRays")

set(NBL_EXT_RADEON_RAYS_H
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/RadeonRays.h
)

set(NBL_EXT_RADEON_RAYS_SRC
	RadeonRays.cpp
)

set(NBL_EXT_RADEON_RAYS_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${NBL_ROOT_PATH}/3rdparty/radeonrays/RadeonRays/include"
	"${NBL_ROOT_PATH}/3rdparty/EGL/EGL/include" # just because it's including COpenGLDriver.h
)

nbl_create_ext_library_project(
	RADEON_RAYS
	"${NBL_EXT_RADEON_RAYS_H}"
	"${NBL_EXT_RADEON_RAYS_SRC}"
	"${NBL_EXT_RADEON_RAYS_EXTERNAL_INCLUDE}"
	""
    ""
)

add_dependencies(NblExtRADEON_RAYS ${RADEON_RAYS_DEPENDENTS})
target_link_libraries(NblExtRADEON_RAYS INTERFACE ${RADEON_RAYS_DEPENDENT_LIBS})