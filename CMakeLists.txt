# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h.in or nabla.h

cmake_minimum_required(VERSION 3.29)

project(Nabla LANGUAGES CXX C)

include(cmake/init/buildSystem.cmake NO_POLICY_SCOPE)

get_filename_component(NBL_ROOT_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
get_filename_component(NBL_ROOT_PATH_BINARY "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
get_filename_component(THIRD_PARTY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/3rdparty" ABSOLUTE)

# TODO reorganize
option(NBL_BUILD_DPL "Enable DPL (Dynamic Parallelism Library)" OFF)
option(NBL_PCH "Enable pre-compiled header" ON)
option(NBL_FAST_MATH "Enable fast low-precision math" ON)
option(NBL_BUILD_EXAMPLES "Enable building examples" ON)
option(NBL_BUILD_DOCS "Enable building documentation?" OFF) # No one has doxygen installed, plus we dont know when was the last time we generated working doxy and we'll use SphinX in the future
option(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION "" ON)
option(NBL_EMBED_BUILTIN_RESOURCES "Embed built-in resources?" ON)
option(NBL_CPACK_INCLUDE_EXAMPLES "CPack with examples and media" ON)
option(NBL_3RDPARTY_FIND_EXPORTED_TARGETS "Do not add ./3rdparty as subdirecotry - look for it's imported targets. It's possible for 3rpdarty project to export all targets if its NBL_3RDPARTY_EXPORT_TO_BUILD_TREE is enabled, for details check it's CMakeLists.txt" OFF)

#include(submodules/update)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# cmake-git-version-tracking
# note that checking if a repository is dirty may cost build time (especially a lot if like us you have a lot of submodules) - by default we run with all checks but if you want to increase build time iterations I recommend to exclude this check
option(GIT_EXCLUDE_IS_DIRTY "Exclude IS_DIRTY from git tracking checks, will increase build time iterations at the expense of the meta information loss" OFF)
add_subdirectory(3rdparty/git-version-tracking EXCLUDE_FROM_ALL)
NBL_ADD_GIT_TRACKING_META_LIBRARY(nabla "${NBL_ROOT_PATH}")
NBL_ADD_GIT_TRACKING_META_LIBRARY(dxc "3rdparty/dxc/dxc")
NBL_GENERATE_GIT_TRACKING_META()

#NBL_3RDPARTY_EXPORT_MODULE_NAME
#NBL_EXTENSIONS_EXPORT_MODULE_NAME

if(NBL_3RDPARTY_FIND_EXPORTED_TARGETS)
	find_file(NBL_3RDPARTY_EXPORT_MODULE
		${NBL_3RDPARTY_EXPORT_MODULE_NAME}.cmake
		PATHS 3rdparty/build ${THIRD_PARTY_BINARY_DIR} ${NBL_3RDPARTY_EXPORT_DIRECTORY} # custom, you can override
		DOC "Filepath to module with exported 3rdparty import targets"
		NO_CACHE
		REQUIRED
		NO_DEFAULT_PATH
		NO_PACKAGE_ROOT_PATH
		NO_CMAKE_PATH
		NO_CMAKE_ENVIRONMENT_PATH
		NO_SYSTEM_ENVIRONMENT_PATH
		NO_CMAKE_SYSTEM_PATH
		NO_CMAKE_INSTALL_PREFIX
	)
	include(${NBL_3RDPARTY_EXPORT_MODULE})
else()
	add_subdirectory(3rdparty EXCLUDE_FROM_ALL)
endif()

add_subdirectory(src/nbl) # Nabla library sources
add_subdirectory(tests) # Python Framework

if(NBL_BUILD_EXAMPLES)
	# /Arek
	# someone did really bad thing when designing this, NOTHING should be outputed to sources NEVER
	# I'm aware of "what about CWD" guh now and shit but this makes git sources dirty (ok you .gitignore) + is a big problem for 
	# containers/build systems which would like to build our codebase in parallel with different setups, the lock thing kinda 
	# protect us but at some point we need to refactor it all - build directory is place for all outputs & currently my disk 
	# I keep sources on is littered with executables & artifacts

	# TMP: TODO
	#file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests" DIRECTORY GUARD PROCESS RESULT_VARIABLE NBL_LOCK TIMEOUT 60)
	#add_subdirectory(examples_tests)
	#file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests" DIRECTORY RELEASE RESULT_VARIABLE NBL_LOCK)
endif()

#add_subdirectory(tools EXCLUDE_FROM_ALL)

if(NBL_BUILD_DOCS)
	add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

add_subdirectory(artifacts EXCLUDE_FROM_ALL) # TODO: remove/refactor, old

include(cpack/package)

# git
find_program(GIT_COMMAND git REQUIRED)

cmake_host_system_information(RESULT CPU_NAME QUERY PROCESSOR_NAME)
cmake_host_system_information(RESULT CPU_DESC QUERY PROCESSOR_DESCRIPTION)

add_custom_command(
	OUTPUT system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "OS: ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "Processor: ${CPU_NAME}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "C Compiler: ${CMAKE_C_COMPILER_ID} ver. ${CMAKE_C_COMPILER_VERSION}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "C Flags: ${CMAKE_C_FLAGS}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "CXX Compiler: ${CMAKE_CXX_COMPILER_ID} ver. ${CMAKE_CXX_COMPILER_VERSION}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "NASM: " >> system_info.txt
	COMMAND ${CMAKE_ASM_NASM_COMPILER} -v >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "VulkanSDK: $ENV{VULKAN_SDK}" >> system_info.txt
	
	COMMENT "Generating ${CMAKE_BINARY_DIR}/system_info.txt"
)

add_custom_command(
	OUTPUT cmake_info.json
	COMMAND ${CMAKE_COMMAND} -E capabilities >> cmake_info.json
	COMMENT "Generating ${CMAKE_BINARY_DIR}/cmake_info.json"
)

set(VKSDK_INFO_COMMAND "$ENV{VULKAN_SDK}/bin/vulkaninfoSDK")
add_custom_command(
	OUTPUT vulkan_info.json
	COMMAND ${VKSDK_INFO_COMMAND} -j -o ${CMAKE_BINARY_DIR}/vulkan_info.json
	COMMENT "Generating ${CMAKE_BINARY_DIR}/vulkan_info.json"
)

add_custom_command(
	OUTPUT git_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "Commit: " >> git_info.txt
	COMMAND ${GIT_COMMAND} diff --quiet || ${CMAKE_COMMAND} -E echo_append "DIRTY " >> git_info.txt
	COMMAND ${GIT_COMMAND} rev-parse HEAD >> git_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "Branch: " >> git_info.txt
	COMMAND ${GIT_COMMAND} rev-parse --abbrev-ref HEAD >> git_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "Tag: " >> git_info.txt
	COMMAND ${GIT_COMMAND} describe --tags >> git_info.txt
	COMMAND ${GIT_COMMAND} diff --quiet || (${CMAKE_COMMAND} -E echo >> git_info.txt && ${GIT_COMMAND} diff >> git_info.txt)
	COMMENT "Generating ${CMAKE_BINARY_DIR}/git_info.txt"
)

set(SYSTEM_INFO_DEPENDENCIES system_info.txt cmake_info.json git_info.txt vulkan_info.json)
add_custom_target(system_info 
	DEPENDS ${SYSTEM_INFO_DEPENDENCIES}
	COMMAND ${CMAKE_COMMAND} -E tar c build_info.zip --format=zip ${SYSTEM_INFO_DEPENDENCIES}
	COMMAND ${CMAKE_COMMAND} -E rm -- ${SYSTEM_INFO_DEPENDENCIES}
	COMMENT "Generating ${CMAKE_BINARY_DIR}/build_info.zip"
)