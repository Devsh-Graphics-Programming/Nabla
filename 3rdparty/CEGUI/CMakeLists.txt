if (IRR_BUILD_CEGUI)

set(CEGUI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cegui)
set(CEGUI_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR}/build)
set(CEGUI_INSTALL_PATH ${CEGUI_BUILD_PATH}/install)

if (MSVC) # only MSVC needs to download and build dependencies of CEGUI
    set(CEGUI_DEPS_URL "https://sourceforge.net/projects/crayzedsgui/files/CEGUI%20Mk-2%20Dependencies/0.8.x/cegui-deps-0.8.x-src.zip/download")
    set(CEGUI_DEPS_ARCHIVE ${CMAKE_CURRENT_BINARY_DIR}/cegui-deps-0.8.x-src.zip)
    set(CEGUI_DEPS_PATH ${CMAKE_CURRENT_BINARY_DIR}/cegui-cegui-dependencies-0ecdf3a9e49b)
    set(CEGUI_DEPS_BUILD_PATH ${CEGUI_DEPS_PATH}/build)

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
            -DCEGUI_BUILD_GLEW=OFF
            -DCEGUI_BUILD_GLFW=OFF
            -DCEGUI_BUILD_GLM=ON
            -DCEGUI_BUILD_LUA=OFF
            -DCEGUI_BUILD_MINIZIP=OFF
            -DCEGUI_BUILD_PCRE=OFF
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
        ${CEGUI_PATH}/dependencies
    )
    message(STATUS "Installing done")

endif()

message(STATUS "Building CEGUI")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory build
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

execute_process(
    COMMAND ${CMAKE_COMMAND} ${CEGUI_PATH} -DCMAKE_INSTALL_PREFIX=${CEGUI_INSTALL_PATH}
    WORKING_DIRECTORY ${CEGUI_BUILD_PATH}
)

if (MSVC)
    execute_process(
        COMMAND MSBUILD /p:Configuration=Release INSTALL.vcxproj
        WORKING_DIRECTORY ${CEGUI_BUILD_PATH}
    )
else()
    execute_process(
        COMMAND ${CMAKE_COMMAND}
            --build ${CEGUI_BUILD_PATH}
            --target install
            --config Release
            -j ${IRR_PARALLEL_BUILD_JOBS}
    )
endif()

message(STATUS "CEGUI built")

endif()
