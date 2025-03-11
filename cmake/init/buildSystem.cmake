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
# limitations under the License

include_guard(GLOBAL)

cmake_policy(SET CMP0112 NEW)
cmake_policy(SET CMP0141 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0141.html#policy:CMP0141
cmake_policy(SET CMP0118 NEW) # https://cmake.org/cmake/help/latest/policy/CMP0118.html#policy:CMP0118

include("${CMAKE_CURRENT_LIST_DIR}/../common.cmake")

option(NBL_ENABLE_VS_CONFIG_IMPORT "Request import of .vsconfig file with VS components required to build Nabla" OFF)

if(NBL_ENABLE_VS_CONFIG_IMPORT)
	NBL_IMPORT_VS_CONFIG()
endif()

set(CMAKE_CONFIGURATION_TYPES 
	Debug
	Release
	RelWithDebInfo
)

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

# global scope vars
get_filename_component(NBL_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(THIRD_PARTY_SOURCE_DIR "${NBL_ROOT_PATH}/3rdparty")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${NBL_ROOT_PATH}/cmake" CACHE PATH "")
get_filename_component(NBL_PYTHON_MODULE_ROOT_PATH "${NBL_ROOT_PATH}/tests" ABSOLUTE)
set(NBL_BUILTIN_RESOURCES_DIRECTORY_PATH "${NBL_ROOT_PATH}/include/nbl/builtin")
set(NBL_MEDIA_DIRECTORY "${NBL_ROOT_PATH}/examples_tests/media")
get_filename_component(NBL_MEDIA_DIRECTORY_ABS "${NBL_MEDIA_DIRECTORY}" ABSOLUTE)

get_property(NBL_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# needs refactors, should not be handled dat way
#### <-
set(NBL_BUILD_ANDROID OFF) # TODO: WRONG, do it differently , detectgiven toolset, platform etc
include(ExternalProject)
include(toolchains/android/build) # TODO: WRONG, do not hardcode this
### <-

# libraries build type & runtime setup
option(NBL_STATIC_BUILD "" OFF) # ON for static builds, OFF for shared
option(NBL_DYNAMIC_MSVC_RUNTIME "" ON)
option(NBL_SANITIZE_ADDRESS OFF)

set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>")

if(NOT NBL_STATIC_BUILD)
	if(WIN32 AND MSVC) # TODO: needs correcting those checks
		if(NOT NBL_DYNAMIC_MSVC_RUNTIME)
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
set(CMAKE_CODELITE_USE_TARGETS ON)

# standard setup
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11, but is it really needed?

# TODO: check those 2
set(LLVM_USE_CRT_DEBUG MTd)
set(LLVM_USE_CRT_RELEASE MT)

if(NOT NBL_IS_MULTI_CONFIG AND NOT DEFINED CMAKE_BUILD_TYPE)
	message(FATAL_ERROR "With single config generators CMAKE_BUILD_TYPE must be defined!")
endif()

if(WIN32)
	set(_NBL_PLATFORM_WINDOWS_ 1)
elseif(ANDROID)
	set(_NBL_PLATFORM_ANDROID_ 1)
elseif(UNIX)
	set(_NBL_PLATFORM_LINUX_ 1)
endif()

if(NBL_STATIC_BUILD)
	unset(_NBL_SHARED_BUILD_)
else()
	set(_NBL_SHARED_BUILD_ ON)
endif()

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
	
	set(CMAKE_INSTALL_PREFIX "${NBL_ROOT_PATH}/install/${PLATFORM}" CACHE PATH "Install path") # TODO: wait what, check it
endif()

set(NBL_3RDPARTY_EXPORT_MODULE_NAME nabla-3rdparty-export)
set(NBL_EXTENSIONS_EXPORT_MODULE_NAME nabla-extensions-export)

include(adjust/flags)
include(adjust/definitions)
include("${NBL_ROOT_PATH}/src/nbl/builtin/utils.cmake")

if (UNIX)	
	if(NOT ANDROID)
		set(CMAKE_THREAD_PREFER_PTHREAD 1)
	endif()
endif()

if(NOT TARGET Threads::Threads)
	find_package(Threads REQUIRED)
endif()

if(MSVC)
	link_libraries(delayimp)
endif()