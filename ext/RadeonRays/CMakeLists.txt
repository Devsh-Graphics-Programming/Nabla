include(../../cmake/common.cmake)


set(IRR_EXT_RADEON_RAYS_H
	RadeonRays.h
)

set(IRR_EXT_RADEON_RAYS_SRC
	RadeonRays.cpp
)

set(IRR_EXT_RADEON_RAYS_EXTERNAL_INCLUDE
	"${IRR_ROOT_PATH}/3rdparty"
	"${IRR_ROOT_PATH}/3rdparty/radeonrays/RadeonRays/include"
)

irr_create_ext_library_project(
	RADEON_RAYS
	"${IRR_EXT_RADEON_RAYS_H}"
	"${IRR_EXT_RADEON_RAYS_SRC}"
	"${IRR_EXT_RADEON_RAYS_EXTERNAL_INCLUDE}"
	""
)

add_dependencies(IrrExtRADEON_RAYS ${RADEON_RAYS_DEPENDENTS})
target_link_libraries(IrrExtRADEON_RAYS INTERFACE ${RADEON_RAYS_DEPENDENT_LIBS})