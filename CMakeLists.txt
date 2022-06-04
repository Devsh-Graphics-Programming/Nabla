# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

cmake_minimum_required(VERSION 3.16)
#policies
cmake_policy(SET CMP0072 OLD) #not yet, would need to get pointers to all OpenGL functions
set(OpenGL_GL_PREFERENCE LEGACY)
#cmake_policy(SET CMP )

project(Nabla LANGUAGES CXX C)

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
get_filename_component(NBL_BINARY_PATH "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  option(USE_CCACHE "Some dependency is using it, but I just don't know which!" ON)
else(CCACHE_FOUND)
  option(USE_CCACHE "Some dependency is using it, but I just don't know which!" OFF)
endif(CCACHE_FOUND)

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
		add_compile_options(-fno-exceptions)

		add_compile_options(
			-Wextra
			-Wno-unused-parameter
			-Wno-unused-but-set-parameter
			-fno-strict-aliasing
			-msse4.2
			-mfpmath=sse
			-fuse-ld=gold
			
			-Wextra
			-Wno-sequence-point
			-Wno-error=ignored-attributes
			-Wno-error=unused-function
			-Wno-error=unused-variable
			-Wno-error=unused-parameter
			
			#"$<$<CONFIG:DEBUG>:-g>" # same as with -O3 below
			"$<$<CONFIG:DEBUG>:-Wall>"
			"$<$<CONFIG:DEBUG>:-fno-omit-frame-pointer>"
			"$<$<CONFIG:DEBUG>:-fstack-protector-strong>"
			#"$<$<CONFIG:RELEASE>:-O3>" # added by default and apparently cmake is not clever enough to filter duplicates for this one
			"$<$<CONFIG:RELEASE>:-fexpensive-optimizations>"
		)
		if (NBL_SANITIZE_ADDRESS)
			add_compile_options(-fsanitize=address)
		endif()
		if (NBL_GCC_SANITIZE_THREAD)
			add_compile_options(-fsanitize=thread)
		endif()
		if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.1)
			add_compile_options(-Wno-error=ignored-attributes)
		endif()
		
		link_libraries(
			-fuse-ld=gold
		)
		if(FAST_MATH)
			add_compile_options(
				"$<$<CONFIG:RELEASE>:-ffast-math>"
			)
		endif()
		if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
			add_compile_options("$<$<CONFIG:DEBUG>:-ggdb3>")
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
	add_definitions(
		-DPNG_THREAD_UNSAFE_OK
		-DPNG_NO_MMX_CODE
		-DPNG_NO_MNG_FEATURES
		-D_7ZIP_ST
		-DSPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS
	)

	if(WIN32)
		add_definitions(
			-DWIN32
			-D__GNUWIN32__
			-D_CRT_SECURE_NO_DEPRECATE
			-DNOMINMAX
		)
	endif()
endmacro()

if(WIN32)
	set(_NBL_PLATFORM_WINDOWS_ 1)
elseif(UNIX)
	set(_NBL_PLATFORM_LINUX_ 1)
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

option(NBL_RUN_TESTS "Enables CI mode - only for Nabla developers" OFF)
if(NBL_RUN_TESTS)
	find_program(_NBL_IMAGEMAGICK_EXE_ NAMES magick.exe PATHS ENV PATH REQUIRED)
endif()

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

if(NBL_BUILD_EXAMPLES)
	add_subdirectory(examples_tests)
endif()

if(NBL_BUILD_DOCS)
	add_subdirectory(docs)
endif()

add_subdirectory(artifacts)
