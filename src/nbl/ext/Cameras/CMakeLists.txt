include(common)

file(GLOB NBL_EXT_CAMERAS_HEADERS CONFIGURE_DEPENDS
    "${NBL_ROOT_PATH}/include/nbl/ext/Cameras/*.hpp"
)

file(GLOB NBL_EXT_CAMERAS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

file(GLOB NBL_EXT_CAMERAS_LOCAL_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

set_source_files_properties(${NBL_EXT_CAMERAS_LOCAL_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)

nbl_create_ext_library_project(
    Cameras
    "${NBL_EXT_CAMERAS_HEADERS}"
    "${NBL_EXT_CAMERAS_SOURCES};${NBL_EXT_CAMERAS_LOCAL_HEADERS}"
    ""
    ""
    ""
)

nbl_install_file_spec(
    "${NBL_ROOT_PATH}/include/nbl/ext/Cameras/README.md"
    "nbl/ext/Cameras"
)

add_library(Nabla::ext::Cameras ALIAS ${LIB_NAME})
