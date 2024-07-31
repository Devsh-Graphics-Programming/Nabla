# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h.in or nabla.h

cmake_minimum_required(VERSION 3.29)
include(cmake/init/buildSystem NO_POLICY_SCOPE)

project(Nabla LANGUAGES CXX C)
enable_language(C CXX ASM)

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

include(submodules/update)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(common)

# TODO: do not include with NBL_3RDPARTY_USE_EXPORT_FROM_BUILD_TREE
add_subdirectory(3rdparty EXCLUDE_FROM_ALL)

# TODO: find exports
#if(NBL_3RDPARTY_USE_EXPORT_FROM_BUILD_TREE)
#	include("${NBL_3RDPARTY_EXPORT_INCLUDE}")
#endif()

add_subdirectory(src/nbl) # Nabla library sources
add_subdirectory(tests) # Python Framework

if(NBL_BUILD_EXAMPLES)

	# /Arek
	# someone did really bad thing when designing this, NOTHING should be outputed to sources NEVER
	# I'm aware of "what about CWD" guh now and shit but this makes git sources dirty (ok you .gitignore) + is a big problem for 
	# containers/build systems which would like to build our codebase in parallel with different setups, the lock thing kinda 
	# protect us but at some point we need to refactor it all - build directory is place for all outputs & currently my disk 
	# I keep sources on is littered with executables & artifacts

	file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests" DIRECTORY GUARD PROCESS RESULT_VARIABLE NBL_LOCK TIMEOUT 60)
	add_subdirectory(examples_tests)
	file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/examples_tests" DIRECTORY RELEASE RESULT_VARIABLE NBL_LOCK)
endif()

add_subdirectory(tools EXCLUDE_FROM_ALL)

if(NBL_BUILD_DOCS)
	add_subdirectory(docs EXCLUDE_FROM_ALL)
endif()

add_subdirectory(artifacts EXCLUDE_FROM_ALL) # TODO: remove/refactor, old

include(cpack/package)
