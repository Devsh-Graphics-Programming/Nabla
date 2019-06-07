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


set(CEGUI_DEPS_BUILD_PATH ${CMAKE_CURRENT_BINARY_DIR}/dependencies/build)
set(CEGUI_DEPS_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR}/dependencies/install CACHE INTERNAL "")

execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory dependencies/build
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

function(build_CEGUI_dependency _NAME _SOURCE_PATH)
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_NAME}
        WORKING_DIRECTORY ${CEGUI_DEPS_BUILD_PATH}
    )
	if (MSVC)
		set(GENERATOR_PLATFORM_OPTION "-A x64")
	endif()
    execute_process(
        COMMAND ${CMAKE_COMMAND} ${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE_PATH} -G "${CMAKE_GENERATOR}" ${GENERATOR_PLATFORM_OPTION} -DCMAKE_INSTALL_PREFIX=${CEGUI_DEPS_INSTALL_PATH} ${ARGN}
        WORKING_DIRECTORY ${CEGUI_DEPS_BUILD_PATH}/${_NAME}
    )
    execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${CEGUI_DEPS_BUILD_PATH}/${_NAME} --target install --config Release -j  ${IRR_PARALLEL_BUILD_JOBS}
        WORKING_DIRECTORY ${CEGUI_DEPS_BUILD_PATH}/${_NAME}
    )
endfunction()

message(STATUS "Building and Pseudo-Installing CEGUI 0.8.x dependencies (because CEGUI's CMake just has to rely on find_package)")
build_CEGUI_dependency(EXPAT ../libexpat/expat "-DBUILD_tools=OFF"  "-DBUILD_examples=OFF"  "-DBUILD_tests=OFF" "-DBUILD_shared=OFF" "-DBUILD_doc=OFF" "-DUSE_libbsd=OFF" "-DMSVC_USE_STATIC_CRT=ON")
if (IRR_FAST_MATH)
	build_CEGUI_dependency(GLM ../glm "-DBUILD_SHARED_LIBS=OFF"  "-DBUILD_STATIC_LIBS=OFF"  "-DGLM_TEST_ENABLE_CXX_14=ON" "-DGLM_TEST_ENABLE_FAST_MATH=ON" "-DGLM_TEST_ENABLE_SIMD_SSE4_2=ON" "-DGLM_TEST_ENABLE=OFF")
else()
	build_CEGUI_dependency(GLM ../glm "-DBUILD_SHARED_LIBS=OFF"  "-DBUILD_STATIC_LIBS=OFF"  "-DGLM_TEST_ENABLE_CXX_14=ON" "-DGLM_TEST_ENABLE_SIMD_SSE4_2=ON" "-DGLM_TEST_ENABLE=OFF")
endif()
build_CEGUI_dependency(Freetype ../freetype2 "-DBUILD_SHARED_LIBS=OFF" "-DCMAKE_DISABLE_FIND_PACKAGE_HarfBuzz=TRUE" "-DCMAKE_DISABLE_FIND_PACKAGE_BZip2=TRUE")
build_CEGUI_dependency(PCRE ./pcre "-DPCRE_SUPPORT_UTF=ON" "-DBUILD_SHARED_LIBS=OFF" "-DPCRE_BUILD_PCREGREP=OFF" "-DPCRE_BUILD_TESTS=OFF" "-DBUILD_doc=OFF")
build_CEGUI_dependency(GLEW ./glew/build/cmake "-Dglew-cmake_BUILD_SHARED=OFF" "-Dglew-cmake_BUILD_MULTI_CONTEXT=OFF")
#build_CEGUI_dependency(GLEW ./glew/build/cmake "-Dglew-cmake_BUILD_SHARED=OFF" "-DUSE_GLEW=OFF")
message(STATUS "CEGUI 0.8.x dependencies built and pseudo-installed")


# Disable unused CEGUI options
set(CEGUI_INSTALL_PKGCONFIG OFF CACHE BOOL "")
set(CEGUI_BUILD_APPLICATION_TEMPLATES OFF CACHE BOOL "")
set(CEGUI_BUILD_PYTHON_MODULES OFF CACHE BOOL "")
set(CEGUI_BUILD_LUA_MODULE OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_XERCES OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_TINYXML OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_TINYXML2 OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_LIBXML2 OFF CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_RAPIDXML OFF CACHE BOOL "")
set(CEGUI_BUILD_RENDERER_OPENGL OFF CACHE BOOL "")
set(CEGUI_USE_FRIBIDI OFF CACHE BOOL "") # enable when Hebrew and Arabic users complain
set(CEGUI_USE_MINIBIDI OFF CACHE BOOL "")
set(CEGUI_SAMPLES_ENABLED OFF CACHE BOOL "")

# Enable required CEGUI options
set(CEGUI_STRING_CLASS "2" CACHE STRING "")
set(CEGUI_HAS_FREETYPE ON CACHE BOOL "")
set(CEGUI_HAS_PCRE_REGEX ON CACHE BOOL "")
set(CEGUI_BUILD_XMLPARSER_EXPAT ON CACHE BOOL "")
set(CEGUI_BUILD_IMAGECODEC_STB ON CACHE BOOL "") # needs at least 1 image codec module
set(CEGUI_BUILD_RENDERER_OPENGL3 ON CACHE BOOL "")
set(CEGUI_BUILD_STATIC_CONFIGURATION ON CACHE BOOL "")
set(CEGUI_BUILD_STATIC_FACTORY_MODULE ON CACHE BOOL "")
set(CEGUI_STATIC_XMLPARSER_MODULE "CEGUIExpatParser" CACHE STRING "")


set(CEGUI_STATIC_DEPENDENCIES ON CACHE BOOL "")
set(CEGUI_BUILD_SHARED_LIBS_WITH_STATIC_DEPENDENCIES ON CACHE BOOL "") # util someone patches CEGUI to define GLEW_STATIC when the above is enabled as well


# override find_package
set(_OLD_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
set(CMAKE_PREFIX_PATH ${CEGUI_DEPS_INSTALL_PATH})
find_package(EXPAT REQUIRED)
find_package(Freetype REQUIRED)
find_package(GLEW REQUIRED)
find_package(PCRE REQUIRED)
add_subdirectory(cegui EXCLUDE_FROM_ALL)
set(CMAKE_PREFIX_PATH ${_OLD_CMAKE_PREFIX_PATH})
