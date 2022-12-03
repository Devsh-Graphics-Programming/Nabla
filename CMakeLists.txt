# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h.in or nabla.h

cmake_minimum_required(VERSION 3.25)
#policies
cmake_policy(SET CMP0072 OLD) #not yet, would need to get pointers to all OpenGL functions
set(OpenGL_GL_PREFERENCE LEGACY)
#cmake_policy(SET CMP )

set(NBL_BUILD_ANDROID OFF)

include(ExternalProject)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchains/android/build.cmake)

project(Nabla LANGUAGES CXX C)

option(NBL_STATIC_BUILD "" ON) # ON for static builds, OFF for shared
option(NBL_DYNAMIC_MSVC_RUNTIME "" OFF)
option(NBL_SANITIZE_ADDRESS OFF)

if(NBL_STATIC_BUILD)
	message(STATUS "Static Nabla build enabled!")
else()
	if(WIN32 AND MSVC)
		if(NBL_DYNAMIC_MSVC_RUNTIME)
			message(STATUS "Shared Nabla build enabled!")
		else()
			message(FATAL_ERROR "Turn NBL_DYNAMIC_MSVC_RUNTIME on! For dynamic Nabla builds dynamic MSVC runtime is mandatory!")
		endif()
	else()
		message(FATAL_ERROR "Nabla can't be built with shared libraries! Please make sure you are targetting Windows OS and MSVC compiler!")
	endif()
endif()

if(NBL_BUILD_ANDROID)
	enable_language(ASM)
endif()

find_package(Vulkan)
if (Vulkan_FOUND)
	message(STATUS "Building with Vulkan driver")
else()
	message(STATUS "Vulkan driver is not enabled")
endif()

option(NBL_COMPILE_WITH_CUDA "Compile with CUDA interop?" OFF)

if(NBL_COMPILE_WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
	if(${CUDAToolkit_VERSION} VERSION_GREATER "9.0")
		message(STATUS "CUDA version 9.0+ found!")
	else()
		message(FATAL_ERROR "CUDA version 9.0+ needed for C++14 support!")
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

set(CMAKE_CXX_STANDARD 20)
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

if(NBL_SANITIZE_ADDRESS) # get rid of RTC1 flag provided by CMake for Debug and RelWithDebInfo configurations for ASAN purposes
	unset(CMAKE_CXX_FLAGS_DEBUG CACHE) 
	unset(CMAKE_CXX_FLAGS_DEBUG)
	unset(CMAKE_CXX_FLAGS_RELWITHDEBINFO CACHE) 
	unset(CMAKE_CXX_FLAGS_RELWITHDEBINFO)
endif()

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()

if (DEFINED ENV{NUMBER_OF_PROCESSORS})
	set(NBL_PARALLEL_BUILD_JOBS $ENV{NUMBER_OF_PROCESSORS})
else()
	set(NBL_PARALLEL_BUILD_JOBS 8)
endif()

macro(nbl_adjust_flags)
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		option(NBL_SANITIZE_THREAD OFF)
	endif()

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

		add_compile_options(-fno-exceptions)

		if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

			add_compile_options(
			-Wno-unused-but-set-parameter	
			$<$<CONFIG:RELEASE>:-fexpensive-optimizations>
			)
		endif()
		if (NBL_SANITIZE_ADDRESS)
			add_compile_options(-fsanitize=address) 
		endif()
		if (NBL_SANITIZE_THREAD)
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
		if(NBL_SANITIZE_ADDRESS)
			set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /wholearchive:clang_rt.asan_dbg-x86_64.lib /wholearchive:clang_rt.asan_cxx_dbg-x86_64.lib -fno-omit-frame-pointer -fsanitize=address /fsanitize=address")
			set(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} /wholearchive:clang_rt.asan-x86_64.lib /wholearchive:clang_rt.asan_cxx-x86_64.lib -fno-omit-frame-pointer -fsanitize=address /fsanitize=address")
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

if(NBL_STATIC_BUILD)
	unset(_NBL_SHARED_BUILD_ CACHE)
else()
	set(_NBL_SHARED_BUILD_ ON CACHE INTERNAL "")
endif()

#uncomment in the future
#option(NBL_TARGET_ARCH_ARM "Build for ARM?" OFF)

option(NBL_COMPILE_WITH_OPENGL "Compile with OpenGL backend?" ON)

option(NBL_PCH "Enable pre-compiled header" ON)

option(NBL_FAST_MATH "Enable fast low-precision math" ON)

option(NBL_BUILD_EXAMPLES "Enable building examples" ON)

option(NBL_BUILD_TOOLS "Enable building tools (just convert2BAW as for now)" ON)

option(NBL_BUILD_MITSUBA_LOADER "Enable nbl::ext::MitsubaLoader?" OFF) # TODO: once it compies turn this ON by default!

option(NBL_BUILD_OPTIX "Enable nbl::ext::OptiX?" OFF)
if(NBL_COMPILE_WITH_CUDA)
	find_package(OPTIX REQUIRED)
	message(STATUS "CUDA enabled and OptiX found!")
else()
	if(NBL_BUILD_OPTIX)
		message(FATAL_ERROR "You cannot build Optix without enabled CUDA! NBL_COMPILE_WITH_CUDA must be ON!")
	endif()
endif()

option(NBL_BUILD_BULLET "Enable Bullet Physics building and integration?" OFF)

option(NBL_BUILD_DOCS "Enable building documentation?" OFF) # No one has doxygen installed, plus we dont know when was the last time we generated working doxy and we'll use SphinX in the future

set(THIRD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(THIRD_PARTY_BINARY_DIR "${PROJECT_BINARY_DIR}/3rdparty")

option(NBL_UPDATE_GIT_SUBMODULE "Turn this ON to let CMake update all public submodules for you" ON)
option(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE "Turn this ON to attempt to update private Nabla submodules" OFF)
find_package(Git REQUIRED)
include(common)

NBL_UPDATE_SUBMODULES()
add_subdirectory(${THIRD_PARTY_SOURCE_DIR})

OPTION(NBL_EMBED_BUILTIN_RESOURCES "Embed built-in resources?" ON)

add_subdirectory(src/nbl) # Nabla code	|	Moved cmakelists.txt from source/Nabla to src/nbl

if(NBL_BUILD_TOOLS)
	add_subdirectory(tools) # convert2BAW
endif()

if(NBL_BUILD_EXAMPLES)
	add_subdirectory(examples_tests)
endif()

if(NBL_BUILD_DOCS)
	add_subdirectory(docs)
endif()

add_subdirectory(artifacts)

export(TARGETS ${_NBL_3RDPARTY_TARGETS_} Nabla NAMESPACE Nabla:: APPEND FILE ${NBL_ROOT_PATH_BINARY}/NablaExport.cmake)
