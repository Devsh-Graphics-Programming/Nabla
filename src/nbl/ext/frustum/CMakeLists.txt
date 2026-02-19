# Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_FRUSTUM_H
    ${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/Frustum/CDrawFrustum.h
)

set(NBL_EXT_FRUSTUM_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/CDrawFrustum.cpp"
)

nbl_create_ext_library_project(
    FRUSTUM
    "${NBL_EXT_FRUSTUM_H}"
    "${NBL_EXT_FRUSTUM_SRC}"
    "${NBL_EXT_FRUSTUM_EXTERNAL_INCLUDE}"
    ""
    ""
)

get_filename_component(_ARCHIVE_ABSOLUTE_ENTRY_PATH_ "${NBL_EXT_INTERNAL_INCLUDE_DIR}" ABSOLUTE)

set(NBL_FRUSTUM_HLSL_MOUNT_POINT "${_ARCHIVE_ABSOLUTE_ENTRY_PATH_}/nbl/ext/Frustum/builtin/hlsl")
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
    ${NBL_FRUSTUM_HLSL_MOUNT_POINT}/common.hlsl
    ${NBL_FRUSTUM_HLSL_MOUNT_POINT}/draw_frustum.unified.hlsl
)
target_sources(${LIB_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

set(SM 6_8)
set(JSON [=[
[
    {
        "INPUT": "${NBL_FRUSTUM_HLSL_MOUNT_POINT}/draw_frustum.unified.hlsl",
        "KEY": "draw_frustum",
    }
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -I "${NBL_ROOT_PATH}/include" # workaround for frustum ext common header which is not part of Nabla builtin archive
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -T lib_${SM}
)

NBL_CREATE_NSC_COMPILE_RULES(
    TARGET ${LIB_NAME}SPIRV
    LINK_TO ${LIB_NAME}
    DEPENDS ${DEPENDS}
    BINARY_DIR ${OUTPUT_DIRECTORY}
    MOUNT_POINT_DEFINE NBL_FRUSTUM_HLSL_MOUNT_POINT
    COMMON_OPTIONS ${COMPILE_OPTIONS}
    OUTPUT_VAR KEYS
    INCLUDE nbl/ext/Frustum/builtin/build/spirv/keys.hpp
    NAMESPACE nbl::ext::frustum::builtin::build
    INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
    NAMESPACE nbl::ext::frustum::builtin::build
    TARGET ${LIB_NAME}_builtinsBuild
    LINK_TO ${LIB_NAME}
    BIND ${OUTPUT_DIRECTORY}
    BUILTINS ${KEYS}
)

add_library(Nabla::ext::Frustum ALIAS ${LIB_NAME})
