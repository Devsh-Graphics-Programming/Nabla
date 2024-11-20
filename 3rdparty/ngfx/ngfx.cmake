option(NBL_BUILD_WITH_NGFX "Enable NGFX build" OFF)

# NOTE: on windows default installation path is:
# "C:/Program Files/NVIDIA Corporation/Nsight Graphics <version>/SDKs/NsightGraphicsSDK" <- define as "NGFX_SDK" environment variable
# then you can pick SDK version with "NGFX_SDK_VERSION" cache variable (CMake GUI list supported)

if(NBL_BUILD_WITH_NGFX)
    if(NOT DEFINED ENV{NGFX_SDK})
        message(FATAL_ERROR "\"NGFX_SDK\" environment variable must be defined to build with NBL_BUILD_WITH_NGFX enabled!")
    endif()

    set(NGFX_SDK "$ENV{NGFX_SDK}")
    cmake_path(NORMAL_PATH NGFX_SDK OUTPUT_VARIABLE NGFX_SDK)

    if(NOT EXISTS "${NGFX_SDK}")
        message(FATAL_ERROR "Found \"NGFX_SDK\" environment variable but it is invalid, env:NGFX_SDK=\"${NGFX_SDK}\" doesn't exist!")
    endif()

    file(GLOB ENTRIES "${NGFX_SDK}/*")

    set(NGFX_VERSIONS "")
    foreach(ENTRY ${ENTRIES})
        if(IS_DIRECTORY ${ENTRY})
            list(APPEND NGFX_VERSIONS ${ENTRY})
        endif()
    endforeach()

    if(NOT NGFX_VERSIONS)
        message(FATAL_ERROR "Could not find any NGFX SDK Version!")
    endif()

    list(TRANSFORM NGFX_VERSIONS REPLACE "${NGFX_SDK}/" "")
    list(SORT NGFX_VERSIONS)
    list(GET NGFX_VERSIONS -1 LATEST_NGFX_VERSION)

    # on the cache variable init pick the latest version, then let user pick from list
    set(NGFX_SDK_VERSION "${LATEST_NGFX_VERSION}" CACHE STRING "NGFX SDK Version")
    set_property(CACHE NGFX_SDK_VERSION PROPERTY STRINGS ${NGFX_VERSIONS})

    set(NGFX_SDK_BASE "${NGFX_SDK}/$CACHE{NGFX_SDK_VERSION}")

    # TODO: wanna support more *host* platforms? (*)
    # NOTE: also I'm hardcoding windows x64 library requests till I know the answer for (*)
    find_file(NBL_NGFX_INJECTION_HEADER NGFX_Injection.h PATHS ${NGFX_SDK_BASE}/include)
    find_file(NBL_NGFX_INJECTION_DLL NGFX_Injection.dll PATHS ${NGFX_SDK_BASE}/lib/x64)
    find_file(NBL_NGFX_INJECTION_IMPORT_LIBRARY NGFX_Injection.lib PATHS ${NGFX_SDK_BASE}/lib/x64)

    if(NBL_NGFX_INJECTION_HEADER AND NBL_NGFX_INJECTION_DLL AND NBL_NGFX_INJECTION_IMPORT_LIBRARY)
        message(STATUS "Enabled build with NVIDIA Nsight Graphics SDK $CACHE{NGFX_SDK_VERSION}\nlocated in: \"${NGFX_SDK_BASE}\"")
    else()
        message(STATUS "Could not enable build with NVIDIA Nsight Graphics SDK $CACHE{NGFX_SDK_VERSION} - invalid components!")
        message(STATUS "Located in: \"${NGFX_SDK_BASE}\"")
        message(STATUS "NBL_NGFX_INJECTION_HEADER=\"${NBL_NGFX_INJECTION_HEADER}\"")
        message(STATUS "NBL_NGFX_INJECTION_DLL=\"${NBL_NGFX_INJECTION_DLL}\"")
        message(STATUS "NBL_NGFX_INJECTION_IMPORT_LIBRARY=\"${NBL_NGFX_INJECTION_IMPORT_LIBRARY}\"")
        message(FATAL_ERROR "You installation may be corupted, please fix it and re-run CMake or disable NBL_BUILD_WITH_NGFX!")
    endif()

    add_library(ngfx INTERFACE)
    target_sources(ngfx INTERFACE "${NBL_NGFX_INJECTION_HEADER}")
    target_include_directories(ngfx INTERFACE "${NGFX_SDK_BASE}/include")
    target_link_libraries(ngfx INTERFACE "${NBL_NGFX_INJECTION_IMPORT_LIBRARY}")
endif()