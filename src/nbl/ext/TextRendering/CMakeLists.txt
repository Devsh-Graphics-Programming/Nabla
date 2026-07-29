include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_TEXT_RENDERING_H
	# extra headers goes there
	# eg. ${NBL_EXT_INTERNAL_INCLUDE_DIR}/something.hpp
)

set(NBL_EXT_TEXT_RENDERING_SRC
	TextRendering.cpp
)

set(NBL_EXT_TEXT_RENDERING_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	"${NBL_EXT_INTERNAL_INCLUDE_DIR}"
)

nbl_create_ext_library_project(
	TEXT_RENDERING
	"${NBL_EXT_TEXT_RENDERING_H}"
	"${NBL_EXT_TEXT_RENDERING_SRC}"
	"${NBL_EXT_TEXT_RENDERING_EXTERNAL_INCLUDE}"
	""
    ""
)