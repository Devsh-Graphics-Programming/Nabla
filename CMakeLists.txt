# Copyright (c) 2019 DevSH Graphics Programming Sp. z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


cmake_minimum_required(VERSION 3.14)
#policies
#cmake_policy(SET CMP0072 NEW) #not yet, would need to get pointers to all OpenGL functions
#cmake_policy(SET CMP )

project(Irrlicht LANGUAGES CXX C)

# CUDA test
include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 9.0)
	message(STATUS "CUDA not found or version lower than 9.0 found, version 9.0+ needed for C++14 support.")
else()
	message(STATUS "CUDA version 9.0+ found, you can enable the IRR_COMPILE_WITH_CUDA option.")
endif()

option(IRR_COMPILE_WITH_CUDA "Compile with CUDA interop?" OFF)
if (IRR_COMPILE_WITH_CUDA)
	enable_language(CUDA)
endif()


#
get_filename_component(IRR_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)

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

if((MSVC_VERSION AND NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 19.11) OR
  (CMAKE_COMPILER_IS_GNUCXX AND NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 7.0) OR
  (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.9))
    set(CMAKE_CXX_STANDARD 17)
else()
    set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Cause CUDA is a special snowflake
if (IRR_COMPILE_WITH_CUDA)
	set(CMAKE_CUDA_STANDARD 14)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11, but is it really needed?

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release")
set(LLVM_USE_CRT_DEBUG MTd CACHE STRING "IrrBAW developer note: No idea what subfolder provides this")
set(LLVM_USE_CRT_RELEASE MT CACHE STRING "IrrBAW developer note: No idea what subfolder provides this")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

if (DEFINED ENV{NUMBER_OF_PROCESSORS})
	set(IRR_PARALLEL_BUILD_JOBS $ENV{NUMBER_OF_PROCESSORS})
else()
	set(IRR_PARALLEL_BUILD_JOBS 8)
endif()

macro(irr_adjust_flags)
	option(IRR_GCC_SANITIZE_ADDRESS OFF)
	option(IRR_GCC_SANITIZE_THREAD OFF)

	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		if (NOT IRR_BUILD_CEGUI)
			add_compile_options(-fno-exceptions)
		endif()

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
		if (IRR_GCC_SANITIZE_ADDRESS)
			add_compile_options(-fsanitize=address)
		endif()
		if (IRR_GCC_SANITIZE_THREAD)
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
		# debug
		string(REPLACE "/MDd" "/MTd" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		string(REPLACE "/W3" "/W0" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
		set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zc:__cplusplus /MP /fp:fast")
		string(REPLACE "/MDd" "/MTd" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
		string(REPLACE "/W3" "/W0" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
		set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /MP /fp:fast")
		
		#release
		string(REPLACE "/GS" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}") # for some reason simply replacing /GS -> /GS- doesn't work... so it vanishes here and appears a few lines below!
		string(REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
		string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
		set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /GL /Zc:__cplusplus /MP /Gy- /Zc:wchar_t /Zi /sdl- /GF /Ot /GS- /Oy /Oi /fp:fast")
		string(REPLACE "/GS" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		string(REPLACE "/O2" "/Ox" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		string(REPLACE "/MD" "/MT" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
		set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /GL /MP /Gy- /Zc:wchar_t /Zi /sdl- /GF /Ot /GS- /Oy /Oi /fp:fast")
	else()
		message(WARNING "UNTESTED COMPILER DETECTED, EXPECT WRONG OPTIMIZATION FLAGS! SUBMIT ISSUE ON GITHUB https://github.com/buildaworldnet/IrrlichtBAW/issues")
	endif()
endmacro()

macro(irr_adjust_definitions)
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
	set(_IRR_PLATFORM_WINDOWS_ 1)
elseif(UNIX)
	set(_IRR_PLATFORM_LINUX_ 1)
endif()

#uncomment in the future
#option(IRR_TARGET_ARCH_ARM "Build for ARM?" OFF)

option(IRR_COMPILE_WITH_OPENGL "Compile with OpenGL backend?" ON)
option(IRR_COMPILE_WITH_OPENCL "Compile with OpenCL interop?" ON)

option(IRR_PCH "Enable pre-compiled header" ON)

option(IRR_FAST_MATH "Enable fast low-precision math" ON)

option(IRR_BUILD_EXAMPLES "Enable building examples" ON)

option(IRR_BUILD_TOOLS "Enable building tools (just convert2BAW as for now)" ON)

option(IRR_BUILD_MITSUBA_LOADER "Enable irr::ext::MitsubaLoader?" ON)

option(IRR_BUILD_RADEON_RAYS "Enable irr::ext::RadeonRays?" OFF)

if (IRR_COMPILE_WITH_CUDA)
	find_package(OPTIX)
	if (OPTIX_FOUND)
		message(STATUS "CUDA and OptiX found, you can enable IRR_BUILD_OPTIX")
	else()
		message(STATUS "OptiX not found")
	endif()
	option(IRR_BUILD_OPTIX "Enable irr::ext::OptiX?" OFF)
else()
	message(STATUS "Because IRR_COMPILE_WITH_CUDA is not enabled, the IRR_BUILD_OPTIX option will be hidden.")
endif()

option(IRR_BUILD_CEGUI "Enable CEGUI building and integration?" OFF)

option(IRR_BUILD_BULLET "Enable Bullet Physics building and integration?" OFF)

option(IRR_BUILD_DOCS "Enable building documentation?" ON)

set(THIRD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(THIRD_PARTY_BINARY_DIR "${PROJECT_BINARY_DIR}/3rdparty")

execute_process(COMMAND git submodule init
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
execute_process(COMMAND git submodule sync --recursive
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(${THIRD_PARTY_SOURCE_DIR})

add_subdirectory(source/Irrlicht) # Irrlicht code

if(IRR_BUILD_TOOLS)
	add_subdirectory(tools) # convert2BAW
endif()

add_subdirectory(ext)

if(IRR_BUILD_EXAMPLES)
	add_subdirectory(examples_tests)
endif()

if(IRR_BUILD_DOCS)
	add_subdirectory(docs)
endif()
