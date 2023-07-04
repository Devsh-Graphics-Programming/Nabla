include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include/nbl/ext/ImGui")

set(NBL_EXT_IMGUI_H
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/ImGui.h
)

set(NBL_EXT_IMGUI_SRC
	ImGui.cpp
)

set(NBL_EXT_IMGUI_EXTERNAL_INCLUDE
	"${NBL_ROOT_PATH}/3rdparty"
	# "${IMGUI_INCLUDE_DIR}"
)

nbl_create_ext_library_project(
	IMGUI_UI
	"${NBL_EXT_IMGUI_H}"
	"${NBL_EXT_IMGUI_SRC}"
	"${NBL_EXT_IMGUI_EXTERNAL_INCLUDE}"
	""
    ""
)
