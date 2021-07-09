# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

cmake_minimum_required(VERSION 3.20)
#policies
cmake_policy(SET CMP0072 OLD) #not yet, would need to get pointers to all OpenGL functions
set(OpenGL_GL_PREFERENCE LEGACY)
#cmake_policy(SET CMP )

option(NBL_BUILD_ANDROID "Build for Android?" OFF)

if(DEFINED CMAKE_TOOLCHAIN_FILE) # TODO - validate paths due to CMake path-policy (errors when no path passed in)
	cmake_path(GET CMAKE_TOOLCHAIN_FILE FILENAME _NBL_CMAKE_TOOLCHAIN_FILENAME_)

	if(${_NBL_CMAKE_TOOLCHAIN_FILENAME_} STREQUAL "android.toolchain.cmake")
		message(STATUS "Found Android toolchain! Validating...")
		
		set(_NBL_SUPPORTED_NDK_ "android-ndk-r22b") # supported NDK version
		set(_NBL_SUPPORTED_ABI_ "x86_64")			# supported ABI
		
		# quering ${_NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_}/${_NBL_SUPPORTED_NDK_}/build/cmake/android.toolchain.cmake
		cmake_path(GET CMAKE_TOOLCHAIN_FILE PARENT_PATH _NBL_TOOLCHAIN_NDK_CMAKE_DIRECTORY_)
		cmake_path(GET _NBL_TOOLCHAIN_NDK_CMAKE_DIRECTORY_ PARENT_PATH _NBL_TOOLCHAIN_NDK_BUILD_DIRECTORY_)
		cmake_path(GET _NBL_TOOLCHAIN_NDK_BUILD_DIRECTORY_ PARENT_PATH _NBL_TOOLCHAIN_NDK_ROOT_DIRECTORY_)
		cmake_path(GET _NBL_TOOLCHAIN_NDK_ROOT_DIRECTORY_ PARENT_PATH _NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_)
		
		string(LENGTH ${CMAKE_TOOLCHAIN_FILE} _NBL_CMAKE_TOOLCHAIN_FILE_LENGTH_)
		string(LENGTH ${_NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_} _NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_LENGTH_)
		math(EXPR _NBL_PARENT_NDK_TO_TOOLCHAIN_FILE_LENGTH_ "${_NBL_CMAKE_TOOLCHAIN_FILE_LENGTH_} - ${_NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_LENGTH_}" OUTPUT_FORMAT DECIMAL)
		string(SUBSTRING ${CMAKE_TOOLCHAIN_FILE} ${_NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_LENGTH_} ${_NBL_PARENT_NDK_TO_TOOLCHAIN_FILE_LENGTH_} _NBL_PARENT_NDK_TO_TOOLCHAIN_FILEPATH_)
		
		if(${_NBL_PARENT_NDK_TO_TOOLCHAIN_FILEPATH_} STREQUAL "/${_NBL_SUPPORTED_NDK_}/build/cmake/android.toolchain.cmake")
			message(STATUS "Found Android NDK '${_NBL_SUPPORTED_NDK_}' is appropriate!")
		else()
			message(FATAL_ERROR "Found Android NDK version isn't appropriate. Use '${_NBL_SUPPORTED_NDK_}'!")
		endif()
		
		if(EXISTS "${_NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_}/build-tools")
			if(DEFINED ANDROID_ABI)
				if(${ANDROID_ABI} STREQUAL ${_NBL_SUPPORTED_ABI_})
					message(STATUS "Supported '${_NBL_SUPPORTED_ABI_}' ABI!")
				
				else()
					message(FATAL_ERROR "Found '${ANDROID_ABI}' isn't appropriate. Use '${_NBL_SUPPORTED_ABI_}'!")
				endif()
			else()
				message(STATUS "Could not find 'ANDROID_ABI', using supported ABI: '${_NBL_SUPPORTED_ABI_}'")
				set(ANDROID_ABI ${_NBL_SUPPORTED_ABI_} CACHE INTERNAL "" FORCE)
			endif()
		
			message(STATUS "'${_NBL_SUPPORTED_NDK_}' located in Android SDK path. Building Nabla with Android toolchain!")
		
			set(ANDROID_SDK_ROOT_PATH "${_NBL_EXPECTED_ANDROID_SDK_ROOT_DIRECTORY_}")
			set(ANDROID_NDK_ROOT_PATH ${ANDROID_SDK_ROOT_PATH}/${_NBL_SUPPORTED_NDK_})
				
			set(NBL_BUILD_ANDROID ON)
			set(_NBL_ANDROID_VALIDATED_ ON)
		else()
			message(FATAL_ERROR "${_NBL_SUPPORTED_NDK_} must be placed in Android SDK path!")
		endif()
	endif()
	
	# TODO - another toolchains in future

endif()

if(NBL_BUILD_ANDROID)
	if(NOT DEFINED _NBL_ANDROID_VALIDATED_)
		message(FATAL_ERROR "Android toolchain hasn't been validated properly. Pass an appropriate path to CMAKE_TOOLCHAIN_FILE!")
	endif()

	if(NOT "${CMAKE_GENERATOR}" STREQUAL "Ninja")
		message(FATAL_ERROR "Only Ninja Generator is supported for Android Build!")
	endif()
	
	if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux")
		message(STATUS "Using Linux as a host OS for Android cross-compiling!")
	elseif("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
		message(STATUS "Using Windows as a host OS for Android cross-compiling!")
	else()
		message(FATAL_ERROR "Current host OS for Android cross-compiling isn't allowed!")
	endif()
	
	# TODO - remove those bellow in future
	set(NBL_COMPILE_WITH_SDL2 OFF CACHE BOOL "Off for Android purposes at the moment" FORCE) 			# TODO - compile/linker errors
	set(_NBL_COMPILE_WITH_OPEN_EXR_ OFF CACHE BOOL "Off for Android purposes at the moment" FORCE) 		# TODO - check it out
	set(_NBL_COMPILE_WITH_BAW_LOADER_ OFF CACHE BOOL "Off for Android purposes at the moment" FORCE)	# TODO - remove when BAW-v4 is done
	set(_NBL_COMPILE_WITH_BAW_WRITER_ OFF CACHE BOOL "Off for Android purposes at the moment" FORCE)	# TODO - remove when BAW-v4 is done
	set(_NBL_COMPILE_WITH_OBJ_LOADER_ OFF CACHE BOOL "Off for Android purposes at the moment" FORCE) 	# TODO - references BAW files, linker errors
	set(NBL_BUILD_MITSUBA_LOADER OFF CACHE BOOL "Off for Android purposes at the moment" FORCE) 		# TODO - compile/linker errors
	set(NBL_BUILD_EXAMPLES OFF CACHE BOOL "Off for Android purposes at the moment" FORCE)				# TODO - check it out
	
	find_package(Java 1.8)
	
	if(DEFINED Java_JAVA_EXECUTABLE)
		message(STATUS "Found Java executable!")
	else()
		message(FATAL_ERROR "Could not find java executable for Android build!")
	endif()
	
	string(LENGTH ${Java_JAVA_EXECUTABLE} Java_JAVA_EXECUTABLE_LENGTH)
	set(Java_JAVA_EXE_BIN_LENGTH 13)
	math(EXPR JAVA_HOME_LENGTH "${Java_JAVA_EXECUTABLE_LENGTH} - ${Java_JAVA_EXE_BIN_LENGTH}" OUTPUT_FORMAT DECIMAL)
	string(SUBSTRING ${Java_JAVA_EXECUTABLE} 0 ${JAVA_HOME_LENGTH} JAVA_HOME)

    message(STATUS "Using JAVA_HOME = ${JAVA_HOME}")

    execute_process(COMMAND ${JAVA_HOME}/bin/java -version
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _output
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    
    if(NOT _result AND _output MATCHES "version \"([0-9]+).([0-9]+)")
        message(STATUS "Java in JAVA_HOME is ${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    else()
        message(STATUS "Java in JAVA_HOME is unknown version ${_output} ${_result}")
    endif()

    #option(STRIP_ANDROID_LIBRARY "Strip the resulting android library" OFF)

    if(NOT ANDROID_PLATFORM)
        set(ANDROID_PLATFORM "android-25")
    endif()
	string(SUBSTRING
		"${ANDROID_PLATFORM}"
		8  # length of "android-"
		-1 # take remainder
		ANDROID_API_LEVEL
	)
    
    # default to libc++_static as the other options can cause crashes
    if(NOT ANDROID_STL)
        set(ANDROID_STL "c++_static")
    endif()

    # Choose clang if the NDK has both gcc and clang, since gcc sometimes fails
	# iirc api levels > 21 have only clang available
    set(CMAKE_ANDROID_NDK_TOOLCHAIN_VERSION "clang")

	#
	# android-specific utils and build tools
	#
    set(ANDROID_BUILD_TOOLS_VERSION "" CACHE STRING "Version of Android build-tools to use instead of the default")
	# take most recent version of build tools, if not set explicitely
    if(ANDROID_BUILD_TOOLS_VERSION STREQUAL "")
        file(GLOB __buildTools RELATIVE "${ANDROID_SDK_ROOT_PATH}/build-tools" "${ANDROID_SDK_ROOT_PATH}/build-tools/*")
        list(SORT __buildTools)

        list(GET __buildTools -1 ANDROID_BUILD_TOOLS_VERSION)

        unset(__buildTools)
    endif()
	set(ANDROID_BUILD_TOOLS "${ANDROID_SDK_ROOT_PATH}/build-tools/${ANDROID_BUILD_TOOLS_VERSION}")
	set(ANDROID_JAVA_BIN "${JAVA_HOME}/bin")
	set(ANDROID_JAVA_RT_JAR "${JAVA_HOME}/jre/lib/rt.jar")

	set(ANDROID_APK_TARGET_ID "" CACHE STRING "The Target ID to build the APK for like 'android-99', use <android list targets> to choose another one.")
    if(ANDROID_APK_TARGET_ID STREQUAL "")
        # This seems different from the platform we're targetting,
        # default to the latest available that's greater or equal to our target platform
        file(GLOB __platforms RELATIVE "${ANDROID_SDK_ROOT_PATH}/platforms" "${ANDROID_SDK_ROOT_PATH}/platforms/*")
        list(SORT __platforms)

        # In case we don't find one, target the latest platform
        list(GET __platforms -1 ANDROID_APK_TARGET_ID)

        string(REPLACE "android-" "" __targetPlat "${ANDROID_PLATFORM}")

		# TODO we might want to adjust min version in the future
        # We require at least android 23 for Activity.requestPermissions
        if(__targetPlat LESS 23)
            set(__targetPlat 23)
        endif()

        foreach( __plat ${__platforms})
            string(REPLACE "android-" "" __curPlat "${__plat}")

            if(NOT (__curPlat LESS __targetPlat) )
                set(ANDROID_APK_TARGET_ID "android-${__curPlat}")
                break()
            endif()
        endforeach()

        unset(__platforms)
        unset(__targetPlat)
        unset(__curPlat)
    endif()

    message(STATUS "Using android.jar from platform ${ANDROID_APK_TARGET_ID}")
	set(ANDROID_JAR "${ANDROID_SDK_ROOT_PATH}/platforms/${ANDROID_APK_TARGET_ID}/android.jar")
endif()

project(Nabla LANGUAGES CXX C)

find_package(Vulkan)
if (Vulkan_FOUND)
	message(STATUS "Building with Vulkan driver")
else()
	message(STATUS "Vulkan driver is not enabled")
endif()

option(NBL_COMPILE_WITH_CUDA "Compile with CUDA interop?" OFF)
option(NBL_COMPILE_WITH_SDL2 "Compile with SDL2" OFF)

# CUDA test
include(CheckLanguage)
check_language(CUDA)
#if(NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
if(NOT CMAKE_CUDA_COMPILER)
	if (NBL_COMPILE_WITH_CUDA)
		#message(WARNING "CUDA not found or version (${CMAKE_CUDA_COMPILER_VERSION}) lower than 9.0 found, version 9.0+ needed for C++14 support.")
		message(WARNING "CUDA not found.")
	endif()
else()
	if (NBL_COMPILE_WITH_CUDA)
		enable_language(CUDA)
	else()
		message(STATUS "CUDA version 9.0+ found, you can enable the NBL_COMPILE_WITH_CUDA option.")
	endif()
endif()

get_filename_component(NBL_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(NBL_ROOT_PATH_BINARY "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  option(USE_CCACHE "Some dependency is using it, but I just don't know which!" ON)
else(CCACHE_FOUND)
  option(USE_CCACHE "Some dependency is using it, but I just don't know which!" OFF)
endif(CCACHE_FOUND)
if(USE_CCACHE)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(USE_CCACHE)

# global IDE stuff for examples
set(CMAKE_CODELITE_USE_TARGETS ON CACHE BOOL "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Cause CUDA is a special snowflake
if (NBL_COMPILE_WITH_CUDA)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11, but is it really needed?

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;")
set(LLVM_USE_CRT_DEBUG MTd CACHE STRING "Nabla developer note: No idea what subfolder provides this")
set(LLVM_USE_CRT_RELEASE MT CACHE STRING "Nabla developer note: No idea what subfolder provides this")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

if (DEFINED ENV{NUMBER_OF_PROCESSORS})
	set(NBL_PARALLEL_BUILD_JOBS $ENV{NUMBER_OF_PROCESSORS})
else()
	set(NBL_PARALLEL_BUILD_JOBS 8)
endif()

macro(nbl_adjust_flags)
	option(NBL_SANITIZE_ADDRESS OFF)
	option(NBL_GCC_SANITIZE_THREAD OFF) # @Anastazluk did MSVC finally include the thread sanitizer?

	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		add_compile_options(
			-Wextra
			-Wno-unused-parameter
			-fno-strict-aliasing
			-msse4.2
			-mfpmath=sse
			
			-Wextra
			-Wno-sequence-point
			-Wno-error=ignored-attributes
			-Wno-error=unused-function
			-Wno-error=unused-variable
			-Wno-error=unused-parameter

			$<$<STREQUAL:$<TARGET_PROPERTY:LINKER_LANGUAGE>,C>:-maes>
			
			$<$<CONFIG:DEBUG>:-Wall>
			$<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>
			$<$<CONFIG:DEBUG>:-fstack-protector-strong>
		)
		
		if(ANDROID)
			add_compile_options(
				$<$<CONFIG:DEBUG>:-glldb>
			)
			
			function(NBL_INSERT_ANDROID_FLAGS _NBL_CMAKE_FLAGS_VARIABLE_NAME_)
				string(FIND "${${_NBL_CMAKE_FLAGS_VARIABLE_NAME_}}" "-g -glldb" CHECK)
				if(${CHECK} EQUAL -1)
					set(${_NBL_CMAKE_FLAGS_VARIABLE_NAME_} "${${_NBL_CMAKE_FLAGS_VARIABLE_NAME_}} -g -glldb" CACHE STRING "" FORCE)
				endif()
			endfunction()
			
			NBL_INSERT_ANDROID_FLAGS(CMAKE_ASM_FLAGS_DEBUG)
			NBL_INSERT_ANDROID_FLAGS(CMAKE_CXX_FLAGS_DEBUG)
			NBL_INSERT_ANDROID_FLAGS(CMAKE_C_FLAGS_DEBUG)
			NBL_INSERT_ANDROID_FLAGS(CMAKE_EXE_LINKER_FLAGS_DEBUG)
			NBL_INSERT_ANDROID_FLAGS(CMAKE_MODULE_LINKER_FLAGS_DEBUG)
			NBL_INSERT_ANDROID_FLAGS(CMAKE_SHARED_LINKER_FLAGS_DEBUG)
		else()
			add_compile_options(
				$<$<CONFIG:DEBUG>:-ggdb3>
			)
		endif()

		if (NOT NBL_BUILD_CEGUI)
			add_compile_options(-fno-exceptions)
		endif()

		if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

			add_compile_options(
			-Wno-unused-but-set-parameter	
			$<$<CONFIG:RELEASE>:-fexpensive-optimizations>
			)
		endif()
		if (NBL_SANITIZE_ADDRESS)
			add_compile_options(-fsanitize=address) 
		endif()
		if (NBL_GCC_SANITIZE_THREAD)
			add_compile_options(-fsanitize=thread)
		endif()
		if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.1)
			add_compile_options(-Wno-error=ignored-attributes)
		endif()
		
		# https://github.com/android/ndk/issues/1426
		if (NOT ANDROID)
			link_libraries(
				-fuse-ld=gold
			)
		endif()

		if(FAST_MATH)
			add_compile_options(
				$<$<CONFIG:RELEASE>:-ffast-math>
			)
		endif()

	elseif(MSVC) # /arch:sse3 or anything like this is not needed on x64 on MSVC for enabling sse3 instructions
		if (NBL_SANITIZE_ADDRESS)
			set(NBL_MSVC_FLAG_SANITIZE_ADDRESS "/fsanitize=address")
			message(WARNING "Due to https://github.com/google/sanitizers/issues/328 expect not to be able to use the Debug target!")
			set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /wholearchive:clang_rt.asan_dbg-x86_64.lib /wholearchive:clang_rt.asan_cxx_dbg-x86_64.lib")
			set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /wholearchive:clang_rt.asan-x86_64.lib /wholearchive:clang_rt.asan_cxx-x86_64.lib")
		endif()
		
		# debug
		string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zc:__cplusplus /MP /ZI /fp:fast /Zc:wchar_t /INCREMENTAL ${NBL_MSVC_SANITIZE_ADDRESS}" )
		string(REPLACE "/W3" "/W0" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
		set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP /ZI /fp:fast /Zc:wchar_t /INCREMENTAL ${NBL_MSVC_SANITIZE_ADDRESS}")
		
		# release
		string(REPLACE "/GS" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # for some reason simply replacing /GS -> /GS- doesn't work... so it vanishes here and appears a few lines below!
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /Zc:__cplusplus /MP /Gy- /Zc:wchar_t /sdl- /GF /GS- /fp:fast ${NBL_MSVC_SANITIZE_ADDRESS}")
		string(REPLACE "/GS" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /GL /MP /Gy- /Zc:wchar_t /sdl- /GF /GS- /fp:fast ${NBL_MSVC_SANITIZE_ADDRESS}")
		
		# relWithDebInfo
		string(REPLACE "/GS" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
		set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /GL /Zc:__cplusplus /Zc:wchar_t /MP /Gy /Zi /sdl- /Oy- /fp:fast ${NBL_MSVC_SANITIZE_ADDRESS}")
		string(REPLACE "/GS" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
		set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /GL /MP /Gy /Zc:wchar_t /Zi /sdl- /Oy- /fp:fast ${NBL_MSVC_SANITIZE_ADDRESS}")
		
		#reason for INCREMENTAL:NO: https://docs.microsoft.com/en-us/cpp/build/reference/ltcg-link-time-code-generation?view=vs-2019 /LTCG is not valid for use with /INCREMENTAL.
		set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO /LTCG:incremental")
	else()
		message(WARNING "UNTESTED COMPILER DETECTED, EXPECT WRONG OPTIMIZATION FLAGS! SUBMIT ISSUE ON GITHUB https://github.com/Devsh-Graphics-Programming/Nabla/issues")
	endif()
endmacro()

macro(nbl_adjust_definitions)
	add_compile_definitions(
		PNG_THREAD_UNSAFE_OK
		PNG_NO_MMX_CODE
		PNG_NO_MNG_FEATURES
		_7ZIP_ST
		SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	)
	
	if(ANDROID)
		add_compile_definitions(
			NBL_ANDROID_TOOLCHAIN
		)
	endif()

	if(WIN32)
		add_compile_definitions(
			WIN32
			__GNUWIN32__
			_CRT_SECURE_NO_DEPRECATE
			NOMINMAX
		)
	endif()
endmacro()

if(WIN32)
	set(_NBL_PLATFORM_WINDOWS_ 1)
elseif(ANDROID)
	set(_NBL_PLATFORM_ANDROID_ 1)
elseif(UNIX)
	set(_NBL_PLATFORM_LINUX_ 1)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
endif()

#uncomment in the future
#option(NBL_TARGET_ARCH_ARM "Build for ARM?" OFF)

option(NBL_COMPILE_WITH_OPENGL "Compile with OpenGL backend?" ON)
option(NBL_COMPILE_WITH_OPENCL "Compile with OpenCL interop?" ON)

option(NBL_PCH "Enable pre-compiled header" ON)

option(NBL_FAST_MATH "Enable fast low-precision math" ON)

option(NBL_BUILD_EXAMPLES "Enable building examples" ON)

option(NBL_BUILD_TOOLS "Enable building tools (just convert2BAW as for now)" ON)

option(NBL_BUILD_MITSUBA_LOADER "Enable nbl::ext::MitsubaLoader?" ON)

option(NBL_BUILD_RADEON_RAYS "Enable nbl::ext::RadeonRays?" OFF)

if (NBL_COMPILE_WITH_CUDA)
	find_package(OPTIX)
	if (OPTIX_FOUND)
		message(STATUS "CUDA and OptiX found, you can enable NBL_BUILD_OPTIX")
	else()
		message(STATUS "OptiX not found")
	endif()
	option(NBL_BUILD_OPTIX "Enable nbl::ext::OptiX?" OFF)
else()
	message(STATUS "Because NBL_COMPILE_WITH_CUDA is not enabled, the NBL_BUILD_OPTIX option will be hidden.")
endif()

option(NBL_BUILD_CEGUI "Enable CEGUI building and integration?" OFF)

option(NBL_BUILD_BULLET "Enable Bullet Physics building and integration?" OFF)

option(NBL_BUILD_DOCS "Enable building documentation?" OFF) # No one has doxygen installed, plus we dont know when was the last time we generated working doxy and we'll use SphinX in the future

set(THIRD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(THIRD_PARTY_BINARY_DIR "${PROJECT_BINARY_DIR}/3rdparty")

execute_process(COMMAND git submodule init
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
execute_process(COMMAND git submodule sync --recursive
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(${THIRD_PARTY_SOURCE_DIR})

OPTION(NBL_EMBED_BUILTIN_RESOURCES "Embed built-in resources?" ON)

add_subdirectory(src/nbl) # Nabla code	|	Moved cmakelists.txt from source/Nabla to src/nbl

if(NBL_BUILD_TOOLS)
	add_subdirectory(tools) # convert2BAW
endif()

if(ANDROID)
add_subdirectory(android-sample EXCLUDE_FROM_ALL)
endif()

if(NBL_BUILD_EXAMPLES)
	add_subdirectory(examples_tests)
endif()

if(NBL_BUILD_DOCS)
	add_subdirectory(docs)
endif()

export(TARGETS ${_NBL_3RDPARTY_TARGETS_} Nabla NAMESPACE Nabla:: APPEND FILE ${NBL_ROOT_PATH_BINARY}/NablaExport.cmake)
