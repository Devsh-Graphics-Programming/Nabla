include(${NBL_ROOT_PATH}/cmake/common.cmake)
set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include/irr/ext/RadeonRays")

set(NBL_EXT_RADEON_RAYS_H
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/RadeonRays.h
)

set(NBL_EXT_RADEON_RAYS_SRC
	RadeonRays.cpp
)

set(NBL_EXT_RADEON_RAYS_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${NBL_ROOT_PATH}/3rdparty/radeonrays/RadeonRays/include"
)

irr_create_ext_library_project(
	RADEON_RAYS
	"${NBL_EXT_RADEON_RAYS_H}"
	"${NBL_EXT_RADEON_RAYS_SRC}"
	"${NBL_EXT_RADEON_RAYS_EXTERNAL_INCLUDE}"
	""
)

add_dependencies(IrrExtRADEON_RAYS ${RADEON_RAYS_DEPENDENTS})
target_link_libraries(IrrExtRADEON_RAYS INTERFACE ${RADEON_RAYS_DEPENDENT_LIBS})