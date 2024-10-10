set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_IMGUI_H
	${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/ImGui/ImGui.h
)

set(NBL_EXT_IMGUI_SRC
	"${CMAKE_CURRENT_SOURCE_DIR}/ImGui.cpp"
)

set(NBL_EXT_IMGUI_INCLUDE_SEARCH_DIRECTORIES
	$<TARGET_PROPERTY:imtestengine,INCLUDE_DIRECTORIES>
)

nbl_create_ext_library_project(
	IMGUI_UI
	"${NBL_EXT_IMGUI_H}"
	"${NBL_EXT_IMGUI_SRC}"
	"${NBL_EXT_IMGUI_EXTERNAL_INCLUDE}"
	""
    ""
)

target_link_libraries(${LIB_NAME} PUBLIC imtestengine)

# (*) -> I wish we could just take NSC, offline-precompile to SPIRV, embed into builtin resource library (as we did!) but then be smart & adjust at runtime OpDecorate of our resources according to wishes - unfortunately no linker yet we have and we are not going to make one ourselves so we compile imgui shaders at runtime
set(_BR_TARGET_ extImguibuiltinResourceData)
set(_RESOURCE_DIR_ "shaders")

get_filename_component(_SEARCH_DIRECTORIES_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "common.hlsl")
LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "psinput.hlsl")
LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "vertex.hlsl") # (*) -> this we could precompile [no resources for which set/binding Ixs could be adjusted] but I'm not going to mix stuff
LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "fragment.hlsl") # (*) -> but this we could not since we let users to provide external descriptor set layout + ImGUI textures & sampler state set/binding Ixs (for pipeline layout) at runtime

ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_SEARCH_DIRECTORIES_}" "${_RESOURCE_DIR_}" "nbl::ext::imgui::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")
LINK_BUILTIN_RESOURCES_TO_TARGET(${LIB_NAME} ${_BR_TARGET_})