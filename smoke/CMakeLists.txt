cmake_minimum_required(VERSION 3.30)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# -> we should call find_package but Nabla doesn't output relocatable packages yet
if(NOT DEFINED NBL_INSTALL_PREFIX)
    set(NBL_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/build-ct/install")
endif()

if(NOT EXISTS ${NBL_INSTALL_PREFIX})
    message(FATAL_ERROR 
        "NBL_INSTALL_PREFIX must be set to the Nabla install path\n"
        "NBL_INSTALL_PREFIX = \"${NBL_INSTALL_PREFIX}\""
    )
endif()

add_library(Nabla INTERFACE)
target_link_libraries(Nabla INTERFACE delayimp)

foreach(CONFIG Debug Release RelWithDebInfo)
    string(TOLOWER ${CONFIG} CONFIG_LOWER)

    if(${CONFIG_LOWER} MATCHES release)
        set(ENTRY ${NBL_INSTALL_PREFIX})
    else()
        set(ENTRY ${NBL_INSTALL_PREFIX}/${CONFIG_LOWER})
    endif()

    if(EXISTS ${ENTRY}/include)
        list(APPEND CONFIGURATIONS ${CONFIG})
        file(GLOB NABLA_LIB "${ENTRY}/lib/nbl/*Nabla*.lib")
        file(GLOB NABLA_SHARED "${ENTRY}/runtime/nbl/*Nabla*.dll")
        get_filename_component(SHARED_NAME "${NABLA_SHARED}" NAME)

        target_link_libraries(Nabla INTERFACE $<$<CONFIG:${CONFIG}>:${NABLA_LIB}>)
        target_link_options(Nabla INTERFACE $<$<CONFIG:${CONFIG}>:/DELAYLOAD:${SHARED_NAME}>)
        target_include_directories(Nabla INTERFACE "${ENTRY}/include")

        message(STATUS
            "-- Configuring with Nabla ${CONFIG} interface setup"
        )
    else()
        # ok for CI as we don't have merged install configs into single prefix in pipelines, we build with matrix and have config per axis dispatch
        message(STATUS
            "-- Skipping Nabla ${CONFIG} interface setup due to missing include directory"
        )
    endif()
endforeach()
# <-

set(CMAKE_CONFIGURATION_TYPES ${CONFIGURATIONS})

add_compile_options(
    /Gm-
    /bigobj 
    /Zc:wchar_t
    /Zc:preprocessor
    /Zc:inline
    /Zc:forScope

    $<$<CONFIG:Debug,RelWithDebInfo>:/GF>
)

project(NablaSmoke CXX)
add_executable(smoke main.cpp)
target_link_libraries(smoke PRIVATE Nabla)

set(CMAKE_CTEST_ARGUMENTS --output-on-failure)
enable_testing()

set(OPTS 
    NBL_EXPLICIT_MODULE_LOAD_LOG=1 
    NBL_EXPLICIT_MODULE_REQUEST_LOG=1
    NBL_INSTALL_DIRECTORY=${NBL_INSTALL_PREFIX}
)

add_test(NAME NBL_INSTALL_LOAD_API
	COMMAND "$<TARGET_FILE:smoke>"
)
set_tests_properties(NBL_INSTALL_LOAD_API PROPERTIES ENVIRONMENT "${OPTS}")