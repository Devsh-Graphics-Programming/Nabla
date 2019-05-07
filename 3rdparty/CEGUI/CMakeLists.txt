if (IRR_BUILD_CEGUI)

set(CEGUI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cegui)
set(CEGUI_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR}/build)
set(CEGUI_INSTALL_PATH ${CEGUI_BUILD_PATH}/install)
set(CEGUI_INSTALL_PATH ${CEGUI_INSTALL_PATH} PARENT_SCOPE)

if (MSVC) # only MSVC needs to download and build dependencies of CEGUI
    set(CEGUI_DEPS_URL "https://sourceforge.net/projects/crayzedsgui/files/CEGUI%20Mk-2%20Dependencies/0.8.x/cegui-deps-0.8.x-src.zip/download")
    set(CEGUI_DEPS_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/cegui-deps-0.8.x-src.zip)
    set(CEGUI_DEPS_PATH ${CMAKE_CURRENT_BINARY_DIR}/cegui-cegui-dependencies-0ecdf3a9e49b)
    set(CEGUI_DEPS_BUILD_PATH ${CEGUI_DEPS_PATH}/build)
    set(CEGUI_DEPS_INSTALL_PATH ${CEGUI_PATH}/dependencies)
    set(CEGUI_DEPS_INSTALL_PATH ${CEGUI_DEPS_INSTALL_PATH} PARENT_SCOPE)

    if (NOT EXISTS "${CEGUI_DEPS_ARCHIVE}")
        message(STATUS "Downloading CEGUI 0.8.x dependencies")
        file(
            DOWNLOAD "${CEGUI_DEPS_URL}" "${CEGUI_DEPS_ARCHIVE}"
            SHOW_PROGRESS
        )
    endif()
    message(STATUS "CEGUI 0.8.x dependencies downloaded")

    if (NOT EXISTS "${CEGUI_DEPS_PATH}")
        message(STATUS "Extracting CEGUI 0.8.x dependencies")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf ${CEGUI_DEPS_ARCHIVE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
        message(STATUS "Extracting done")
    endif()

    message(STATUS "Building CEGUI 0.8.x dependencies")

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory build
        WORKING_DIRECTORY ${CEGUI_DEPS_PATH}
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND} ..
            -DCEGUI_BUILD_CORONA=OFF
            -DCEGUI_BUILD_DEVIL=OFF
            -DCEGUI_BUILD_EXPAT=ON
            -DCEGUI_BUILD_FREEIMAGE=OFF
            -DCEGUI_BUILD_FREETYPE2=ON
            -DCEGUI_BUILD_GLEW=ON
            -DCEGUI_BUILD_GLFW=OFF
            -DCEGUI_BUILD_GLM=ON
            -DCEGUI_BUILD_LUA=OFF
            -DCEGUI_BUILD_MINIZIP=OFF
            -DCEGUI_BUILD_PCRE=ON
            -DCEGUI_BUILD_SILLY=OFF
            -DCEGUI_BUILD_TINYXML=OFF
            -DCEGUI_BUILD_TOLUAPP=OFF
            -DCEGUI_BUILD_XERCES=OFF
            -DCEGUI_BUILD_ZLIB=OFF
            -DCEGUI_MSVC_STATIC_RUNTIME=ON
        WORKING_DIRECTORY ${CEGUI_DEPS_BUILD_PATH}
    )

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${CEGUI_DEPS_BUILD_PATH} --config Release -j  ${IRR_PARALLEL_BUILD_JOBS}
    )

    message(STATUS "CEGUI 0.8.x dependencies built")

    message(STATUS "Installing CEGUI 0.8.x dependencies")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CEGUI_DEPS_BUILD_PATH}/dependencies
        ${CEGUI_DEPS_INSTALL_PATH}
    )
    message(STATUS "Installing done")

endif()

message(STATUS "Building CEGUI")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory build
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# CEGUI needs at least 1 image codec to build although we won't use
execute_process(
    COMMAND ${CMAKE_COMMAND} ${CEGUI_PATH}
        -DCMAKE_INSTALL_PREFIX=${CEGUI_INSTALL_PATH}
        -DCEGUI_INSTALL_PKGCONFIG=OFF
        -DCEGUI_BUILD_APPLICATION_TEMPLATES=OFF
        -DCEGUI_HAS_PCRE_REGEX=ON
        -DCEGUI_BUILD_PYTHON_MODULES=OFF
        -DCEGUI_BUILD_LUA_MODULE=OFF
        -DCEGUI_BUILD_XMLPARSER_XERCES=OFF
        -DCEGUI_BUILD_XMLPARSER_TINYXML=OFF
        -DCEGUI_BUILD_XMLPARSER_TINYXML2=OFF
        -DCEGUI_BUILD_XMLPARSER_LIBXML2=OFF
        -DCEGUI_BUILD_XMLPARSER_RAPIDXML=OFF
        -DCEGUI_BUILD_XMLPARSER_EXPAT=ON
        -DCEGUI_OPTION_DEFAULT_XMLPARSER=ExpatParser
        -DCEGUI_BUILD_IMAGECODEC_STB=ON
        -DCEGUI_BUILD_RENDERER_OPENGL=OFF
        -DCEGUI_BUILD_RENDERER_OPENGL3=ON
        -DCEGUI_SAMPLES_ENABLED=OFF
        -DCEGUI_BUILD_STATIC_CONFIGURATION=ON
        -DCEGUI_BUILD_STATIC_FACTORY_MODULE=ON
        -DCEGUI_STATIC_XMLPARSER_MODULE=CEGUIExpatParser
    WORKING_DIRECTORY ${CEGUI_BUILD_PATH}
)

execute_process(
    COMMAND ${CMAKE_COMMAND}
        --build ${CEGUI_BUILD_PATH}
        --target install
        --config Release
        -j ${IRR_PARALLEL_BUILD_JOBS}
)

message(STATUS "CEGUI built")

endif()
