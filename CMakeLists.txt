# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h.in or nabla.h

cmake_minimum_required(VERSION 3.29)
include(cmake/init/buildSystem.cmake NO_POLICY_SCOPE)

project(Nabla LANGUAGES CXX C)

get_filename_component(NBL_ROOT_PATH_BINARY "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
get_filename_component(THIRD_PARTY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/3rdparty" ABSOLUTE)

# TODO reorganize
option(NBL_PCH "Enable pre-compiled header" ON)
option(NBL_FAST_MATH "Enable fast low-precision math" ON)
option(NBL_BUILD_EXAMPLES "Enable building examples" ON)
option(NBL_BUILD_DOCS "Enable building documentation?" OFF) # No one has doxygen installed, plus we dont know when was the last time we generated working doxy and we'll use SphinX in the future
option(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION "" ON)
option(NBL_EMBED_BUILTIN_RESOURCES "Embed built-in resources?" ON)
option(NBL_CPACK_INCLUDE_EXAMPLES "CPack with examples and media" ON)
option(NBL_3RDPARTY_FIND_EXPORTED_TARGETS "Do not add ./3rdparty as subdirecotry - look for it's imported targets. It's possible for 3rpdarty project to export all targets if its NBL_3RDPARTY_EXPORT_TO_BUILD_TREE is enabled, for details check it's CMakeLists.txt" ON)

#include(submodules/update)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(common)

#NBL_3RDPARTY_EXPORT_MODULE_NAME
#NBL_EXTENSIONS_EXPORT_MODULE_NAME

if(NBL_3RDPARTY_FIND_EXPORTED_TARGETS)
	find_file(NBL_3RDPARTY_EXPORT_MODULE
		${NBL_3RDPARTY_EXPORT_MODULE_NAME}.cmake
		PATHS 3rdparty/build ${NBL_3RDPARTY_EXPORT_DIRECTORY}
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
