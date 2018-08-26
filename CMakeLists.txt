cmake_minimum_required(VERSION 3.1)
project(Irrlicht)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) #...without compiler extensions like gnu++11, but is it really needed?

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/")

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type: Debug or Release" FORCE)
endif()

option(FAST_MATH "Enable fast low-precision math" ON)

option(BUILD_EXAMPLES "Enable building examples" ON)

option(BUILD_TOOLS "Enable building tools (just convert2BAW as for now)" ON)

option(NASTY_OPENSSL_WORKAROUND "Do not check this on if you actually want to build Irrlicht; just for cmake scripts testing purposes" OFF)
add_subdirectory(source/Irrlicht) # Irrlicht code

if(BUILD_TOOLS)
	add_subdirectory(tools) # convert2BAW
endif()
if(BUILD_EXAMPLES)
	add_subdirectory(examples_tests)
endif()