# Check if Conan is present in the system
find_program(CONAN_EXECUTABLE conan)
if (NOT CONAN_EXECUTABLE)
    # Conan is not found, install it using PIP
    find_program(PIP_EXECUTABLE pip)
    if (NOT PIP_EXECUTABLE)
        message(FATAL_ERROR "Pip executable not found. Please install python and add it to PATH to proceed with Conan installation.")
    endif()

    # Install conan using pip
    execute_process(
        COMMAND ${PIP_EXECUTABLE} install conan
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE pip_result
    )

    if (NOT pip_result EQUAL "0")
        message(FATAL_ERROR "Failed to install Conan using pip. Please check your Python and pip installation.")
    endif()

    # After installation, try to find Conan again
    find_program(CONAN_EXECUTABLE conan)
    if (NOT CONAN_EXECUTABLE)
        message(FATAL_ERROR "Conan executable not found. Please ensure Conan is in your PATH.")
    endif()
endif()

# Determine the Conan compiler name based on the CMake compiler ID
if (NOT CONAN_COMPILER)
  if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
      set(CONAN_COMPILER "apple-clang")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(CONAN_COMPILER "clang")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(CONAN_COMPILER "msvc")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CONAN_COMPILER "gcc")
  endif()
endif()

# Extract the major version number from the CMake compiler version
if (NOT CONAN_COMPILER_VERSION)
    if (CONAN_COMPILER STREQUAL "msvc")
        message (STATUS "Detected MSVC version: ${CMAKE_CXX_COMPILER_VERSION}")

        # special handling for msvc to extract the major version (e.g., 143 from 14.3.0)
        string (REGEX MATCH "^[0-9]+\\.[0-9]" MSVC_VERSION_MATCH ${CMAKE_CXX_COMPILER_VERSION})

        # remove the dot to get the major version (e.g., 143 from 14.3)
        string (REPLACE "." "" CONAN_COMPILER_VERSION ${MSVC_VERSION_MATCH})
    else()
        string(REGEX MATCH "^[0-9]+" CONAN_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})
    endif()
endif()

# Map Architectures to Conan's expected values
if (NOT CONAN_ARCH)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(CONAN_ARCH "x86_64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i386|i686")
    set(CONAN_ARCH "x86")
  else ()
    message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}, using it directly for Conan")
    set(CONAN_ARCH ${CMAKE_SYSTEM_PROCESSOR})
  endif()
endif()

# Map CXX_STANDARD to Conan's expected value
if (NOT CONAN_CXX_STANDARD)
    if (CMAKE_CXX_STANDARD)
        set(CONAN_CXX_STANDARD ${CMAKE_CXX_STANDARD})
    else()
        message(WARNING "CMAKE_CXX_STANDARD is not set, defaulting to C++20 for Conan profile")
        set(CONAN_CXX_STANDARD 20) # Default to C++17 if not specified
    endif()
endif()

# Determine OS-specific Conan settings
if(WIN32)
    set(CONAN_OS_SPECIFIC "compiler.runtime=dynamic")
    
    # Append runtime version for Clang-cl
    if(MSVC_TOOLSET_VERSION AND NOT CONAN_COMPILER STREQUAL "msvc")
        set (CONAN_OS_SPECIFIC "${CONAN_OS_SPECIFIC}\ncompiler.runtime_version=v${MSVC_TOOLSET_VERSION}")
    endif()

elseif(APPLE)
    # macOS always uses LLVM's libc++
    set(CONAN_OS_SPECIFIC "compiler.libcxx=libc++")

elseif(UNIX)
    # On Linux, determine if Clang was forced to use libc++, otherwise default to libstdc++11
    set(CONAN_LIBCXX "libstdc++11") 
    
    if(CONAN_COMPILER STREQUAL "clang")
        # Check if the developer passed -stdlib=libc++ in the CMake cache or env
        if(CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
            set(CONAN_LIBCXX "libc++")
        endif()
    endif()
    
    set(CONAN_OS_SPECIFIC "compiler.libcxx=${CONAN_LIBCXX}")
endif()

# Set the Conan profile content
set(CONAN_PROFILE_PATH "${CMAKE_BINARY_DIR}/conan_profile.txt")
set(CONAN_PROFILE "[settings]
os=${CMAKE_SYSTEM_NAME}
arch=${CONAN_ARCH}
compiler=${CONAN_COMPILER}
compiler.version=${CONAN_COMPILER_VERSION}
compiler.cppstd=${CONAN_CXX_STANDARD}
${CONAN_OS_SPECIFIC}

[conf]
tools.cmake.cmaketoolchain:generator=${CMAKE_GENERATOR}
")

# Make a message with file contents
message(STATUS "Generating Conan profile at ${CONAN_PROFILE_PATH}...")
message(STATUS "Conan profile content:\n${CONAN_PROFILE}")

file(WRITE ${CONAN_PROFILE_PATH} ${CONAN_PROFILE})

if (CMAKE_GENERATOR MATCHES "Ninja Multi-Config" OR CMAKE_GENERATOR MATCHES "Visual Studio")
    if (CMAKE_CONFIGURATION_TYPES)
        set(CONAN_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
        # remove MinSizeRel if it exists, and map it to Release

    else()
        set(CONAN_CONFIGS "Debug;Release;RelWithDebInfo") 
    endif()

    list(REMOVE_ITEM CONAN_CONFIGS "MinSizeRel")
else()
    if(NOT CMAKE_BUILD_TYPE)
        message(WARNING "CMAKE_BUILD_TYPE is not set, defaulting to Release for Conan.")
        set(CMAKE_BUILD_TYPE "Release")
    endif()
    set(CONAN_CONFIGS ${CMAKE_BUILD_TYPE})
endif()

set(CMAKE_MAP_IMPORTED_CONFIG_MINSIZEREL "Release" CACHE STRING "Fallback to Release dependencies for MinSizeRel" FORCE)

# Run Conan install for each configuration
foreach(CONFIG IN LISTS CONAN_CONFIGS)
    message(STATUS "Running Conan install for build type: ${CONFIG}...")
    
    # Pass build_type dynamically
    set(CONAN_CLI_ARGS 
        "-s:h" "build_type=${CONFIG}"
        "-s:b" "build_type=Release"
    )
    
    # Inject MSVC runtime type dynamically for Windows
    if(WIN32)
        if(CONFIG STREQUAL "Debug")
            set(MSVC_RUNTIME_TYPE "Debug")
        else()
            set(MSVC_RUNTIME_TYPE "Release")
        endif()

        list(APPEND CONAN_CLI_ARGS 
            "-s:h" "compiler.runtime_type=${MSVC_RUNTIME_TYPE}"
            "-s:b" "compiler.runtime_type=Release"
        )
    endif()

    execute_process(
        COMMAND ${CONAN_EXECUTABLE} install ${CMAKE_SOURCE_DIR} 
                --profile:all ${CONAN_PROFILE_PATH} 
                ${CONAN_CLI_ARGS}
                --build=missing
                -cc core.graph:compatibility_mode=optimized
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE conan_result
    )

    if(NOT conan_result EQUAL "0")
        message(FATAL_ERROR "Conan install failed for configuration: ${CONFIG}!")
    endif()
endforeach()

if(NOT conan_result EQUAL "0")
    message(FATAL_ERROR "Conan install failed!")
endif()
