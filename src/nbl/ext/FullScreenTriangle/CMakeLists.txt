include(common)

set(HEADERS
    ${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/FullScreenTriangle/FullScreenTriangle.h
)

set(SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/CFullScreenTriangle.cpp
)

nbl_create_ext_library_project(FULL_SCREEN_TRIANGLE
    "${HEADERS}"
    "${SRCS}"
    ""
    ""
    ""
)

get_filename_component(DIR "${NBL_ROOT_PATH}/include/nbl/builtin/hlsl/ext/FullScreenTriangle" ABSOLUTE)
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(SM 6_8)
set(JSON [=[
[
    {
        "INPUT": "${DIR}/default.vert.hlsl",
        "KEY": "full_screen_triangle_vertex",
    }
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -T vs_${SM}
    -E __nbl__hlsl__ext__FullScreenTriangle__vertex_main
)

NBL_CREATE_NSC_COMPILE_RULES(
    TARGET ${LIB_NAME}SPIRV
    LINK_TO ${LIB_NAME}
    BINARY_DIR ${OUTPUT_DIRECTORY}
    MOUNT_POINT_DEFINE NBL_FULL_SCREEN_TRIANGLE_HLSL_MOUNT_POINT
    COMMON_OPTIONS ${COMPILE_OPTIONS}
    OUTPUT_VAR KEYS
    INCLUDE nbl/ext/FullScreenTriangle/builtin/build/spirv/keys.hpp
    NAMESPACE nbl::ext::FullScreenTriangle::builtin::build
    INPUTS ${JSON}
    GLOB_DIR ${DIR}
)

NBL_CREATE_RESOURCE_ARCHIVE(
    NAMESPACE nbl::ext::FullScreenTriangle::builtin::build
    TARGET ${LIB_NAME}_builtinsBuild
    LINK_TO ${LIB_NAME}
    BIND ${OUTPUT_DIRECTORY}
    BUILTINS ${KEYS}
)

add_library(Nabla::ext::FullScreenTriangle ALIAS ${LIB_NAME})