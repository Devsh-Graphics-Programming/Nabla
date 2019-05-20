if (MSVC) # MSVC needs to download and build dependencies of CEGUI
    set(CEGUI_DEPS_URL "https://sourceforge.net/projects/crayzedsgui/files/CEGUI%20Mk-2%20Dependencies/0.8.x/cegui-deps-0.8.x-src.zip/download")
    set(CEGUI_DEPS_ARCHIVE ${CMAKE_CURRENT_SOURCE_DIR}/cegui-deps-0.8.x-src.zip)
    set(CEGUI_DEPS_PATH ${CMAKE_CURRENT_BINARY_DIR}/cegui-cegui-dependencies-0ecdf3a9e49b)
    set(CEGUI_DEPS_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR}/cegui-cegui-dependencies-0ecdf3a9e49b/build)
    set(CEGUI_DEPS_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cegui/dependencies)
    set(CEGUI_DEPS_INSTALL_PATH ${CEGUI_DEPS_INSTALL_PATH} PARENT_SCOPE)

    if (NOT EXISTS "${CEGUI_DEPS_ARCHIVE}")
        message(STATUS "Downloading CEGUI 0.8.x dependencies")
        file(
            DOWNLOAD "${CEGUI_DEPS_URL}" "${CEGUI_DEPS_ARCHIVE}"
            SHOW_PROGRESS
        )
        message(STATUS "CEGUI 0.8.x dependencies downloaded")
    endif()

    if (NOT EXISTS "${CEGUI_DEPS_PATH}")
        message(STATUS "Extracting CEGUI 0.8.x dependencies")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${CEGUI_DEPS_ARCHIVE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        message(STATUS "Extracting done")
    endif()

    if (NOT EXISTS "${CEGUI_DEPS_INSTALL_PATH}")
        message(STATUS "Building CEGUI 0.8.x dependencies")

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E make_directory build
            WORKING_DIRECTORY ${CEGUI_DEPS_PATH}
        )

        execute_process(
            COMMAND ${CMAKE_COMMAND} ..
                -DCEGUI_BUILD_CORONA=OFF
                -DCEGUI_BUILD_DEVIL=OFF
                -DCEGUI_BUILD_FREEIMAGE=OFF
                -DCEGUI_BUILD_GLFW=OFF
                -DCEGUI_BUILD_LUA=OFF
                -DCEGUI_BUILD_MINIZIP=OFF
                -DCEGUI_BUILD_SILLY=OFF
                -DCEGUI_BUILD_TINYXML=OFF
                -DCEGUI_BUILD_TOLUAPP=OFF
                -DCEGUI_BUILD_XERCES=OFF
                -DCEGUI_BUILD_ZLIB=OFF
                -DCEGUI_BUILD_EXPAT=ON
                -DCEGUI_BUILD_FREETYPE2=ON
                -DCEGUI_BUILD_GLEW=ON
                -DCEGUI_BUILD_GLM=ON
                -DCEGUI_BUILD_PCRE=ON
            WORKING_DIRECTORY ${CEGUI_DEPS_BUILD_PATH}
        )

        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${CEGUI_DEPS_BUILD_PATH} --config Release -j  ${IRR_PARALLEL_BUILD_JOBS}
        )

        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${CEGUI_DEPS_BUILD_PATH} --config Debug -j  ${IRR_PARALLEL_BUILD_JOBS}
        )

        message(STATUS "CEGUI 0.8.x dependencies built")

        message(STATUS "Installing CEGUI 0.8.x dependencies")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CEGUI_DEPS_BUILD_PATH}/dependencies/include
            ${CEGUI_DEPS_INSTALL_PATH}/include
        )
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CEGUI_DEPS_BUILD_PATH}/dependencies/lib/static
            ${CEGUI_DEPS_INSTALL_PATH}/lib
        )
        message(STATUS "Installing done")
    endif()
endif()

# Disable unused CEGUI options
set(CEGUI_INSTALL_PKGCONFIG OFF CACHE BOOL "")
set(CEGUI_BUILD_APPLICATION_TEMPLATES OFF CACHE BOOL "")
set(CEGUI_BUILD_PYTHON_MODULES OFF CACHE BOOL "")
set(CEGUI_BUILD_LUA_MODULE OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_XERCES OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_TINYXML OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_TINYXML2 OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_LIBXML2 OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_RAPIDXML OFF CACHE BOOL "")
set(CEGUI_BUILD_RENDERER_OPENGL OFF CACHE BOOL "")
set(CEGUI_SAMPLES_ENABLED OFF CACHE BOOL "")

# Enable required CEGUI options
set(CEGUI_STRING_CLASS "2" CACHE STRING "")
set(CEGUI_HAS_FREETYPE ON CACHE BOOL "")
set(CEGUI_HAS_PCRE_REGEX ON CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_EXPAT ON CACHE BOOL "")
set(CEGUI_BUILD_IMAGECODEC_STB ON CACHE BOOL "") # needs at least 1 image codec module
set(CEGUI_BUILD_RENDERER_OPENGL3 ON CACHE BOOL "")
set(CEGUI_BUILD_STATIC_CONFIGURATION ON CACHE BOOL "")
set(CEGUI_BUILD_STATIC_FACTORY_MODULE ON CACHE BOOL "")
set(CEGUI_STATIC_XMLPARSER_MODULE "CEGUIExpatParser" CACHE STRING "")

if (MSVC) # static linking on Linux will come later, since find_package only returns dynamic libs for now
    set(CEGUI_STATIC_DEPENDENCIES ON CACHE BOOL "")
endif()

add_subdirectory(cegui EXCLUDE_FROM_ALL)
