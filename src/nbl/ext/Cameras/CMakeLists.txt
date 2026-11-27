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

find_package(glm CONFIG REQUIRED)
find_package(gtl CONFIG REQUIRED)
find_package(Imath CONFIG REQUIRED)
find_package(blake3 CONFIG REQUIRED)
find_package(Boost CONFIG REQUIRED)
target_link_libraries(${LIB_NAME} PRIVATE glm::glm gtl::gtl Imath::Imath BLAKE3::blake3 boost::boost)

# installed from src/nbl as the "Cameras" component of the Nabla package
set_target_properties(${LIB_NAME} PROPERTIES
    EXPORT_NAME "ext::Cameras"
    PUBLIC_HEADER "${NBL_EXT_CAMERAS_HEADERS};${NBL_ROOT_PATH}/include/nbl/ext/Cameras/README.md"
)

add_library(Nabla::ext::Cameras ALIAS ${LIB_NAME})
