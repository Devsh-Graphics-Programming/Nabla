# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h.in or nabla.h

cmake_minimum_required(VERSION 3.29)
cmake_policy(SET CMP0112 NEW)
cmake_policy(SET CMP0141 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0141.html#policy:CMP0141

set(NBL_BUILD_ANDROID OFF)

include(ExternalProject)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/toolchains/android/build.cmake)

project(Nabla LANGUAGES CXX C)
enable_language(C CXX ASM)

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
get_filename_component(NBL_PYTHON_MODULE_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tests" ABSOLUTE)
set(NBL_BUILTIN_RESOURCES_DIRECTORY_PATH "${NBL_ROOT_PATH}/include/nbl/builtin")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NBL_ROOT_PATH}/cmake" CACHE PATH "")
get_property(NBL_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

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

set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;")
set(LLVM_USE_CRT_DEBUG MTd CACHE STRING "Nabla developer note: No idea what subfolder provides this")
set(LLVM_USE_CRT_RELEASE MT CACHE STRING "Nabla developer note: No idea what subfolder provides this")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type: Debug or Release" FORCE)
endif()

option(NBL_MEMORY_CONSUMPTION_CHECK_SKIP "Turn it ON to bypass memory consumption test given _NBL_JOBS_AMOUNT_. Be aware you are doing it on your own risk of potential build failures!" ON)
option(NBL_JOBS_OVERRIDE "Override jobs with safe bias if required to respect currently available memory" OFF)

include(ProcessorCount)

if(NOT DEFINED _NBL_JOBS_AMOUNT_)
	if(NOT "$ENV{_NBL_JOBS_AMOUNT_}" STREQUAL "")
		set(_NBL_JOBS_AMOUNT_ "$ENV{_NBL_JOBS_AMOUNT_}")
	else()
		ProcessorCount(_NBL_JOBS_AMOUNT_)
		if(_NBL_JOBS_AMOUNT_ EQUAL 0)
			set(_NBL_JOBS_AMOUNT_ 1)
		endif()
	endif()
endif()

cmake_host_system_information(RESULT _NBL_TOTAL_PHYSICAL_MEMORY_ QUERY TOTAL_PHYSICAL_MEMORY) # MB
cmake_host_system_information(RESULT _NBL_AVAILABLE_PHYSICAL_MEMORY_ QUERY AVAILABLE_PHYSICAL_MEMORY) # MB

if(NBL_JOBS_OVERRIDE)
	math(EXPR _CI_NBL_JOBS_AMOUNT_ "(${_NBL_AVAILABLE_PHYSICAL_MEMORY_} - 512)/(2*1024)") # override with safe bias, respect memory and don't take more then max processors we have
	if(_CI_NBL_JOBS_AMOUNT_ LESS _NBL_JOBS_AMOUNT_)
		message(WARNING "Overriding _NBL_JOBS_AMOUNT_: \"${_NBL_JOBS_AMOUNT_}\" with \"${_CI_NBL_JOBS_AMOUNT_}\"")
	
		set(_NBL_JOBS_AMOUNT_ "${_CI_NBL_JOBS_AMOUNT_}")
	endif()
endif()

message(STATUS "_NBL_JOBS_AMOUNT_: \"${_NBL_JOBS_AMOUNT_}\"")

math(EXPR _NBL_DEBUG_MEMORY_CONSUPTION_WITH_ALL_JOBS_ "${_NBL_JOBS_AMOUNT_}*2*1024") # MB
math(EXPR _NBL_CURRENTLY_USED_PHYSICAL_MEMORY_ "${_NBL_TOTAL_PHYSICAL_MEMORY_}-${_NBL_AVAILABLE_PHYSICAL_MEMORY_}") # MB

if(_NBL_AVAILABLE_PHYSICAL_MEMORY_ LESS_EQUAL _NBL_DEBUG_MEMORY_CONSUPTION_WITH_ALL_JOBS_) # TODO: we may also add Release and RWDI checks as well
	if(NBL_MEMORY_CONSUMPTION_CHECK_SKIP)
		set(_NBL_CMAKE_STATUS_ WARNING)
	else()
		set(_NBL_CMAKE_STATUS_ FATAL_ERROR)
	endif()
	
	message(${_NBL_CMAKE_STATUS_} "Memory consumption issue detected! To protect you from compile and linker errors, please read this message.\n\nYour total physical memory is ${_NBL_TOTAL_PHYSICAL_MEMORY_} MBs, your OS is currently using ${_NBL_CURRENTLY_USED_PHYSICAL_MEMORY_} MBs and consumption of your memory with requested ${_NBL_JOBS_AMOUNT_} jobs in Debug configuration may be around ${_NBL_DEBUG_MEMORY_CONSUPTION_WITH_ALL_JOBS_} MBs. Please override '_NBL_JOBS_AMOUNT_' variable by setting it as cache variable and lower the jobs! If you want to continue anyway, please define 'NBL_MEMORY_CONSUMPTION_CHECK_SKIP' but be aware - you are doing it on your own risk of possible build failures.")
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

option(NBL_BUILD_DPL "Enable DPL (Dynamic Parallelism Library)" OFF)
option(NBL_PCH "Enable pre-compiled header" ON)
option(NBL_FAST_MATH "Enable fast low-precision math" ON)
option(NBL_BUILD_EXAMPLES "Enable building examples" ON)
option(NBL_BUILD_MITSUBA_LOADER "Enable nbl::ext::MitsubaLoader?" OFF) # TODO: once it compies turn this ON by default!
option(NBL_BUILD_IMGUI "Enable nbl::ext::ImGui?" OFF)

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
option(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION "" ON)
option(NBL_EMBED_BUILTIN_RESOURCES "Embed built-in resources?" ON)

set(THIRD_PARTY_SOURCE_DIR "${PROJECT_SOURCE_DIR}/3rdparty")
set(THIRD_PARTY_BINARY_DIR "${PROJECT_BINARY_DIR}/3rdparty")

set(NBL_MEDIA_DIRECTORY "${PROJECT_SOURCE_DIR}/examples_tests/media")
get_filename_component(NBL_MEDIA_DIRECTORY_ABS "${NBL_MEDIA_DIRECTORY}" ABSOLUTE)

include(submodules/update)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(common)

add_subdirectory(${THIRD_PARTY_SOURCE_DIR})

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

export(TARGETS ${_NBL_3RDPARTY_TARGETS_} Nabla NAMESPACE Nabla:: APPEND FILE ${NBL_ROOT_PATH_BINARY}/NablaExport.cmake)
