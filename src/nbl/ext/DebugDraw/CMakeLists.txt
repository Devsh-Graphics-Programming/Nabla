include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_DEBUG_DRAW_H
    ${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/DebugDraw/CDrawAABB.h
)

set(NBL_EXT_DEBUG_DRAW_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/CDrawAABB.cpp"
)

nbl_create_ext_library_project(
    DEBUG_DRAW
    "${NBL_EXT_DEBUG_DRAW_H}"
    "${NBL_EXT_DEBUG_DRAW_SRC}"
    "${NBL_EXT_DEBUG_DRAW_EXTERNAL_INCLUDE}"
    ""
    ""
)

# this should be standard for all extensions
set(_ARCHIVE_ENTRY_KEY_ "DebugDraw/builtin/hlsl") # then each one has unique archive key
get_filename_component(_ARCHIVE_ABSOLUTE_ENTRY_PATH_ "${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)

target_compile_definitions(${LIB_NAME} PRIVATE _ARCHIVE_ABSOLUTE_ENTRY_PATH_="${_ARCHIVE_ABSOLUTE_ENTRY_PATH_}")
target_compile_definitions(${LIB_NAME} PRIVATE _ARCHIVE_ENTRY_KEY_="${_ARCHIVE_ENTRY_KEY_}")

if(NBL_EMBED_BUILTIN_RESOURCES)
    set(_BR_TARGET_ extDebugDrawbuiltinResourceData)

    LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "common.hlsl")
    LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "single.vertex.hlsl") # (*)
    LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "aabb_instances.vertex.hlsl") # (*)
    LIST_BUILTIN_RESOURCE(RESOURCES_TO_EMBED "aabb_instances.fragment.hlsl") # (*)

    ADD_CUSTOM_BUILTIN_RESOURCES(${_BR_TARGET_} RESOURCES_TO_EMBED "${_ARCHIVE_ABSOLUTE_ENTRY_PATH_}" "${_ARCHIVE_ENTRY_KEY_}" "nbl::ext::debug_draw::builtin" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}")
    LINK_BUILTIN_RESOURCES_TO_TARGET(${LIB_NAME} ${_BR_TARGET_})
endif()

set(NBL_EXT_DEBUG_DRAW_TARGET ${LIB_NAME} CACHE INTERNAL "Nabla's Debug Draw logical target name")
