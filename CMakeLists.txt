# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h.in or nabla.h

cmake_minimum_required(VERSION 3.29)
cmake_policy(SET CMP0112 NEW)
cmake_policy(SET CMP0141 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0141.html#policy:CMP0141
cmake_policy(SET CMP0118 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0118.html#policy:CMP0118

project(Nabla LANGUAGES CXX C)
enable_language(C CXX ASM)

get_filename_component(NBL_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(NBL_ROOT_PATH_BINARY "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
get_filename_component(NBL_PYTHON_MODULE_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests" ABSOLUTE)
set(NBL_BUILTIN_RESOURCES_DIRECTORY_PATH "${NBL_ROOT_PATH}/include/nbl/builtin")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NBL_ROOT_PATH}/cmake" CACHE PATH "")
get_property(NBL_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

set(NBL_BUILD_ANDROID OFF) # TODO: WRONG, do it differently , detectgiven toolset, platform etc
include(ExternalProject)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchains/android/build.cmake) # TODO: WRONG, do not hardcode this

option(NBL_STATIC_BUILD "" OFF) # ON for static builds, OFF for shared
option(NBL_DYNAMIC_MSVC_RUNTIME "" ON)
option(NBL_SANITIZE_ADDRESS OFF)

if(MSVC)
	if(NBL_SANITIZE_ADDRESS)
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>")
	else()
		set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug>:EditAndContinue>$<$<CONFIG:RelWithDebInfo>:ProgramDatabase>")
	endif()
endif()

if(NBL_STATIC_BUILD)
	message(STATUS "Static Nabla build enabled!")
else()
	# TODO: needs correcting those checks
	if(WIN32 AND MSVC)
		if(NBL_DYNAMIC_MSVC_RUNTIME)
			message(STATUS "Shared Nabla build enabled!")
		else()
			message(FATAL_ERROR "Turn NBL_DYNAMIC_MSVC_RUNTIME on! For dynamic Nabla builds dynamic MSVC runtime is mandatory!")
		endif()
	endif()
endif()

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
set(CMAKE_CODELITE_USE_TARGETS ON CACHE INTERNAL "")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11, but is it really needed?

set(CMAKE_CONFIGURATION_TYPES 
	Debug
	Release
	RelWithDebInfo
)

set(LLVM_USE_CRT_DEBUG MTd CACHE INTERNAL "Nabla developer note: No idea what subfolder provides this")
set(LLVM_USE_CRT_RELEASE MT CACHE INTERNAL "Nabla developer note: No idea what subfolder provides this")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()

include(adjust/flags)
include(adjust/definitions)

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

# TODO reorganize
option(NBL_PCH "Enable pre-compiled header" ON)
option(NBL_FAST_MATH "Enable fast low-precision math" ON) # Nabla only thing
option(NBL_BUILD_EXAMPLES "Enable building examples" ON)
option(NBL_BUILD_DOCS "Enable building documentation?" OFF) # No one has doxygen installed, plus we dont know when was the last time we generated working doxy and we'll use SphinX in the future
option(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION "" ON)
option(NBL_EMBED_BUILTIN_RESOURCES "Embed built-in resources?" ON)

set(THIRD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(THIRD_PARTY_BINARY_DIR "${PROJECT_BINARY_DIR}/3rdparty")

set(NBL_MEDIA_DIRECTORY "${PROJECT_SOURCE_DIR}/examples_tests/media")
get_filename_component(NBL_MEDIA_DIRECTORY_ABS "${NBL_MEDIA_DIRECTORY}" ABSOLUTE)

include(submodules/update)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(common)

# TODO MOVE TO INCLUDE FILE
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(PLATFORM NOTFOUND)
	if (WIN32)
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
			set(PLATFORM win64-clang)
		elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
			set(PLATFORM win64-gcc)
		elseif (MSVC)
			set(PLATFORM win64-msvc)
		endif()
	elseif (ANDROID)
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
			set(PLATFORM android-clang)
		elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
			set(PLATFORM android-gcc)
		endif()
	elseif (UNIX AND NOT APPLE)
		if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
			set(PLATFORM linux-clang)
		elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
			set(PLATFORM linux-gcc)
		endif()
	endif()
	
	if ("${PLATFORM}" STREQUAL "NOTFOUND")
		message(FATAL_ERROR "Unsupported compiler!")
	endif()
	
	set(CMAKE_INSTALL_PREFIX "${NBL_ROOT_PATH}/install/${PLATFORM}" CACHE PATH "Install path" FORCE)
endif()

# OK: with NBL_3RDPARTY_USE_EXPORT_FROM_BUILD_TREE we need to configure 3rdparty separately, build & reuse its export here

add_subdirectory(${THIRD_PARTY_SOURCE_DIR})

if(NBL_3RDPARTY_USE_EXPORT_FROM_BUILD_TREE)
	include("${NBL_3RDPARTY_EXPORT_INCLUDE}")
endif()

add_subdirectory(src/nbl)
add_subdirectory("${NBL_PYTHON_MODULE_ROOT_PATH}" tests) # Python Framework
if(NBL_BUILD_EXAMPLES)
	file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests" DIRECTORY GUARD PROCESS RESULT_VARIABLE NBL_LOCK TIMEOUT 60)
	add_subdirectory(examples_tests)
	file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests" DIRECTORY RELEASE RESULT_VARIABLE NBL_LOCK)
endif()
add_subdirectory(tools)

if(NBL_BUILD_DOCS)
	add_subdirectory(docs)
endif()

add_subdirectory(artifacts)

option(NBL_CPACK_INCLUDE_EXAMPLES "CPack with examples and media" ON)
include(cpack/package)
