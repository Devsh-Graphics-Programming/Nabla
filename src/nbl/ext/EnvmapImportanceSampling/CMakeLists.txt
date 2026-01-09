include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_H
    ${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/EnvmapImportanceSampling/CEnvmapImportanceSampling.h
)

set(NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/CEnvmapImportanceSampling.cpp"
)

nbl_create_ext_library_project(
    ENVMAP_IMPORTANCE_SAMPLING
    "${NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_H}"
    "${NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_SRC}"
    "${NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_EXTERNAL_INCLUDE}"
    ""
    ""
)

get_filename_component(_ARCHIVE_ABSOLUTE_ENTRY_PATH_ "${NBL_EXT_INTERNAL_INCLUDE_DIR}" ABSOLUTE)

set(NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT "${_ARCHIVE_ABSOLUTE_ENTRY_PATH_}/nbl/ext/EnvmapImportanceSampling/builtin/hlsl")
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/common.hlsl
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/gen_warpmap.comp.hlsl
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/gen_luma.comp.hlsl
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/measure_luma.comp.hlsl
)
target_sources(${LIB_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

set(SM 6_8)
set(JSON [=[
[
    {
        "INPUT": "${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/gen_warpmap.comp.hlsl",
        "KEY": "gen_warpmap",
    },
    {
        "INPUT": "${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/gen_luma.comp.hlsl",
        "KEY": "gen_luma",
    },
    {
        "INPUT": "${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/measure_luma.comp.hlsl",
        "KEY": "measure_luma",
    }
    
]
]=])
string(CONFIGURE "${JSON}" JSON)

set(COMPILE_OPTIONS
    -I "${NBL_ROOT_PATH}/include" # a workaround due to envmap importance sampling ext common header which is not part of Nabla builtin archive
    -I "${CMAKE_CURRENT_SOURCE_DIR}"
    -T lib_${SM}
)

NBL_CREATE_NSC_COMPILE_RULES(
    TARGET ${LIB_NAME}SPIRV
    LINK_TO ${LIB_NAME}
    DEPENDS ${DEPENDS}
    BINARY_DIR ${OUTPUT_DIRECTORY}
    MOUNT_POINT_DEFINE NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT
    COMMON_OPTIONS ${COMPILE_OPTIONS}
    OUTPUT_VAR KEYS
    INCLUDE nbl/ext/EnvmapImportanceSampling/builtin/build/spirv/keys.hpp
    NAMESPACE nbl::ext::envmap_importance_sampling::builtin::build
    INPUTS ${JSON}
)

NBL_CREATE_RESOURCE_ARCHIVE(
    NAMESPACE nbl::ext::envmap_importance_sampling::builtin::build
    TARGET ${LIB_NAME}_builtinsBuild
    LINK_TO ${LIB_NAME}
    BIND ${OUTPUT_DIRECTORY}
    BUILTINS ${KEYS}
)


add_library(Nabla::ext::EnvmapImportanceSampling ALIAS ${LIB_NAME})
