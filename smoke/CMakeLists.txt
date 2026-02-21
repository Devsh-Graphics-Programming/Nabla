cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CONFIGURATION_TYPES Release Debug RelWithDebInfo)

add_compile_options(
    /Gm-
    /bigobj 
    /Zc:wchar_t
    /Zc:preprocessor
    /Zc:inline
    /Zc:forScope

    $<$<CONFIG:Debug,RelWithDebInfo>:/GF>
)

set(CMAKE_SYSTEM_VERSION 10.0)
project(NablaSmoke CXX)
include(${CMAKE_CURRENT_LIST_DIR}/NablaSmokeTests.cmake)

# default hint for our CI, normally it needs to be path to package's directory where all autogen config .cmake scripts are
set(PACKAGE_CONFIG_SEARCH_PATHS ${CMAKE_CURRENT_LIST_DIR}/build-ct/install/cmake ${PACKAGE_CONFIG_SEARCH_PATH_HINTS})
set(NABLA_FIND_PACKAGE_VERBOSE ON)
find_package(Nabla REQUIRED CONFIG 
    PATHS ${PACKAGE_CONFIG_SEARCH_PATHS}
)

add_executable(smoke main.cpp pch.hpp cdb.ps1)
target_link_libraries(smoke PRIVATE Nabla::Nabla)
target_compile_definitions(smoke PRIVATE _AFXDLL)
target_precompile_headers(smoke PRIVATE pch.hpp)

set(NBL_SMOKE_FLOW "NO_BUILD_COPY" CACHE STRING "Smoke runtime flow: NO_BUILD_COPY or WITH_BUILD_COPY")
set_property(CACHE NBL_SMOKE_FLOW PROPERTY STRINGS NO_BUILD_COPY WITH_BUILD_COPY)
string(TOUPPER "${NBL_SMOKE_FLOW}" NBL_SMOKE_FLOW)
message(STATUS "Smoke runtime flow: ${NBL_SMOKE_FLOW}")
option(NBL_SMOKE_INSTALL_SELFTEST "Install smoke with CTest metadata and run tests from install tree" ON)

if(NBL_SMOKE_FLOW STREQUAL "NO_BUILD_COPY")
    # No build-time copy, install-time runtime modules in ./Libraries.
    nabla_setup_runtime_install_modules(smoke
        RUNTIME_MODULES_SUBDIR "Libraries"
    )
elseif(NBL_SMOKE_FLOW STREQUAL "WITH_BUILD_COPY")
    # Build-time copy + install-time runtime modules in ./Libraries.
    nabla_setup_runtime_modules(smoke
        RUNTIME_MODULES_SUBDIR "Libraries"
        INSTALL_RULES ON
    )
else()
    message(FATAL_ERROR "Invalid NBL_SMOKE_FLOW='${NBL_SMOKE_FLOW}'")
endif()

set(CMAKE_CTEST_ARGUMENTS --verbose)
enable_testing()

set(NBL_SMOKE_TEST_ENVIRONMENT 
    NBL_EXPLICIT_MODULE_LOAD_LOG=1 
    NBL_EXPLICIT_MODULE_REQUEST_LOG=1
)

option(ENABLE_CRASH_HANDLER "Enable crash handler" ON)

nabla_smoke_add_install_load_api_test(
    TEST_NAME NBL_INSTALL_LOAD_API
    EXE_PATH "$<TARGET_FILE:smoke>"
    CRASH_HANDLER_SCRIPT "$<SHELL_PATH:${CMAKE_CURRENT_LIST_DIR}/cdb.ps1>"
    ENABLE_CRASH_HANDLER ${ENABLE_CRASH_HANDLER}
    ENVIRONMENT "${NBL_SMOKE_TEST_ENVIRONMENT}"
)

if(NBL_SMOKE_INSTALL_SELFTEST)
    include(GNUInstallDirs)

    install(TARGETS smoke
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/cdb.ps1"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    set(_nbl_smoke_install_cmake_dir "${CMAKE_INSTALL_DATADIR}/nabla-smoke")
    install(FILES "${CMAKE_CURRENT_LIST_DIR}/NablaSmokeTests.cmake"
        DESTINATION "${_nbl_smoke_install_cmake_dir}"
    )

    set(NBL_SMOKE_INSTALL_CMAKE_DIR "${_nbl_smoke_install_cmake_dir}")
    configure_file(
        "${CMAKE_CURRENT_LIST_DIR}/CTestTestfile.install.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.install.cmake"
        @ONLY
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/CTestTestfile.install.cmake"
        DESTINATION "."
        RENAME "CTestTestfile.cmake"
    )
endif()
