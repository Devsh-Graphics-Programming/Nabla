include(common)

file(GLOB NBL_EXT_CAMERAS_HEADERS CONFIGURE_DEPENDS
    "${NBL_ROOT_PATH}/include/nbl/ext/Cameras/*.hpp"
)

set(NBL_EXT_CAMERAS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/CCameraPersistence.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/CCameraScriptedRuntimePersistence.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/CCameraJsonPersistenceUtilities.hpp"
)

set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/CCameraJsonPersistenceUtilities.hpp"
    PROPERTIES
    HEADER_FILE_ONLY ON
)

nbl_create_ext_library_project(
    Cameras
    "${NBL_EXT_CAMERAS_HEADERS}"
    "${NBL_EXT_CAMERAS_SOURCES}"
    ""
    ""
    ""
)

nbl_install_file_spec(
    "${NBL_ROOT_PATH}/include/nbl/ext/Cameras/README.md"
    "nbl/ext/Cameras"
)

add_library(Nabla::ext::Cameras ALIAS ${LIB_NAME})
