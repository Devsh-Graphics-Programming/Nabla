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

add_executable(smoke_override main.cpp pch.hpp cdb.ps1)
target_link_libraries(smoke_override PRIVATE Nabla::Nabla)
target_compile_definitions(smoke_override PRIVATE
    _AFXDLL
    "NBL_CPACK_PACKAGE_NABLA_DLL_DIR=\"./Libraries\""
    "NBL_CPACK_PACKAGE_DXC_DLL_DIR=\"./Libraries\""
)
target_precompile_headers(smoke_override PRIVATE pch.hpp)
set_target_properties(smoke smoke_override PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${Nabla_ROOT}/debug/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${Nabla_ROOT}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${Nabla_ROOT}/relwithdebinfo/bin"
)

set(CMAKE_CTEST_ARGUMENTS --verbose)
enable_testing()

set(OPTS 
    NBL_EXPLICIT_MODULE_LOAD_LOG=1 
    NBL_EXPLICIT_MODULE_REQUEST_LOG=1
)

option(ENABLE_CRASH_HANDLER "Enable crash handler" ON)

if(WIN32)
    if(ENABLE_CRASH_HANDLER)
        set(CMD
            powershell -NoProfile -ExecutionPolicy Bypass 
            -File "$<SHELL_PATH:${CMAKE_CURRENT_LIST_DIR}/cdb.ps1>" 
            -Exe "$<TARGET_FILE:smoke>"
        )
    endif()
endif()

if(NOT ENABLE_CRASH_HANDLER)
    set(CMD
        "$<TARGET_FILE:smoke>"
    )
endif()

add_test(NAME NBL_INSTALL_LOAD_API COMMAND ${CMD})
set_tests_properties(NBL_INSTALL_LOAD_API PROPERTIES ENVIRONMENT "${OPTS}")

set(_SMOKE_CPACK_REL_ENTRY "./$<$<NOT:$<STREQUAL:$<CONFIG>,Release>>:$<LOWER_CASE:$<CONFIG>>>")
set(_SMOKE_NABLA_RUNTIME_DIR "${Nabla_ROOT}/${_SMOKE_CPACK_REL_ENTRY}/runtime/nbl")
set(_SMOKE_DXC_RUNTIME_DIR "${_SMOKE_NABLA_RUNTIME_DIR}/3rdparty/dxc")
set(_SMOKE_OVERRIDE_LIB_DIR "$<TARGET_FILE_DIR:smoke_override>/Libraries")

add_test(NAME NBL_INSTALL_LOAD_API_OVERRIDE
    COMMAND "${CMAKE_COMMAND}"
        "-DSMOKE_EXE=$<TARGET_FILE:smoke_override>"
        "-DNABLA_RUNTIME_DIR=${_SMOKE_NABLA_RUNTIME_DIR}"
        "-DDXC_RUNTIME_DIR=${_SMOKE_DXC_RUNTIME_DIR}"
        "-DDXC_MODULE_BASENAME=dxcompiler"
        "-DOVERRIDE_DIR=${_SMOKE_OVERRIDE_LIB_DIR}"
        "-P" "$<SHELL_PATH:${CMAKE_CURRENT_LIST_DIR}/run_override_test.cmake>"
)
set_tests_properties(NBL_INSTALL_LOAD_API_OVERRIDE PROPERTIES ENVIRONMENT "${OPTS}")
