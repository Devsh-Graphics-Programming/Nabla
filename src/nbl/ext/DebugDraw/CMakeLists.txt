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

get_filename_component(_ARCHIVE_ABSOLUTE_ENTRY_PATH_ "${NBL_EXT_INTERNAL_INCLUDE_DIR}" ABSOLUTE)

set(NBL_DEBUG_DRAW_HLSL_MOUNT_POINT "${_ARCHIVE_ABSOLUTE_ENTRY_PATH_}/nbl/ext/DebugDraw/builtin/hlsl")
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
    ${NBL_DEBUG_DRAW_HLSL_MOUNT_POINT}/common.hlsl
    ${NBL_DEBUG_DRAW_HLSL_MOUNT_POINT}/single.vertex.hlsl
    ${NBL_DEBUG_DRAW_HLSL_MOUNT_POINT}/aabb_instances.vertex.hlsl
    ${NBL_DEBUG_DRAW_HLSL_MOUNT_POINT}/aabb_instances.fragment.hlsl
)
target_sources(${LIB_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

target_compile_definitions(${LIB_NAME} PRIVATE _ARCHIVE_ABSOLUTE_SPV_PATH_="${OUTPUT_DIRECTORY}")

set(SM 6_8)
set(JSON [=[
[
    {
        "INPUT": "${NBL_DEBUG_DRAW_HLSL_MOUNT_POINT}/draw_aabb.unified.hlsl",
        "KEY": "draw_aabb",
    }
    
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -T lib_${SM}
)

NBL_CREATE_NSC_COMPILE_RULES(
    TARGET ${LIB_NAME}SPIRV
    LINK_TO ${LIB_NAME}
    DEPENDS ${DEPENDS}
    BINARY_DIR ${OUTPUT_DIRECTORY}
    MOUNT_POINT_DEFINE NBL_DEBUG_DRAW_HLSL_MOUNT_POINT
    COMMON_OPTIONS ${COMPILE_OPTIONS}
    OUTPUT_VAR KEYS
    INCLUDE nbl/ext/DebugDraw/builtin/build/spirv/keys.hpp
    NAMESPACE nbl::ext::debug_draw::builtin::build
    INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
    NAMESPACE nbl::ext::debug_draw::builtin::build
    TARGET ${LIB_NAME}_builtinsBuild
    LINK_TO ${LIB_NAME}
    BIND ${OUTPUT_DIRECTORY}
    BUILTINS ${KEYS}
)


add_library(Nabla::ext::DebugDraw ALIAS ${LIB_NAME})
