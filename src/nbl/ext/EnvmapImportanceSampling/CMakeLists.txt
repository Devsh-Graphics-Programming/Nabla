include(${NBL_ROOT_PATH}/cmake/common.cmake)

set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include")

set(NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_H
    ${NBL_EXT_INTERNAL_INCLUDE_DIR}/nbl/ext/EnvmapImportanceSampling/CEnvmapImportanceSampling.h
)

set(NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/CEnvmapImportanceSampling.cpp"
)

get_filename_component(_ARCHIVE_ABSOLUTE_ENTRY_PATH_ "${NBL_EXT_INTERNAL_INCLUDE_DIR}" ABSOLUTE)

set(NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT "${_ARCHIVE_ABSOLUTE_ENTRY_PATH_}/nbl/ext/EnvmapImportanceSampling/builtin/hlsl")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/common.hlsl
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/gen_luma.comp.hlsl
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/gen_warp.comp.hlsl
    ${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}/measure_luma.comp.hlsl
)

nbl_create_ext_library_project(
    ENVMAP_IMPORTANCE_SAMPLING
    "${NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_H}"
    "${NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_SRC}"
    "${NBL_EXT_ENVMAP_IMPORTANCE_SAMPLING_EXTERNAL_INCLUDE}"
    ""
    NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT="${NBL_ENVMAP_IMPORTANCE_SAMPLING_HLSL_MOUNT_POINT}"
)

target_sources(${LIB_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)


NBL_CREATE_RESOURCE_ARCHIVE(
    NAMESPACE nbl::ext::envmap_importance_sampling::builtin::build
    TARGET ${LIB_NAME}_builtinsBuild
    LINK_TO ${LIB_NAME}
    BIND ${OUTPUT_DIRECTORY}
    BUILTINS
        common.hlsl
        gen_luma.comp.hlsl
        gen_warp.comp.hlsl
        measure_luma.comp.hlsl

)


add_library(Nabla::ext::EnvmapImportanceSampling ALIAS ${LIB_NAME})
