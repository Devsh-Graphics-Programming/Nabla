# Copyright (c) 2022 DevSH Graphics Programming Sp. z O.O.
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

# CMake script for generating msvc manifest during build time

if(DEFINED NBL_ROOT_PATH)
	if(NOT EXISTS ${NBL_ROOT_PATH})
		message(FATAL_ERROR  "NBL_ROOT_PATH as '${NBL_ROOT_PATH}' is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_ROOT_PATH variable must be specified for this script!")
endif()

if(NOT DEFINED NBL_GEN_DIRECTORY)
	message(FATAL_ERROR "NBL_GEN_DIRECTORY variable must be specified for this script!")
endif()

if(NOT DEFINED NABLA_DLL_PATH)
	message(FATAL_ERROR  "NABLA_DLL_PATH variable must be specified for this script!")
endif()

cmake_path(GET NABLA_DLL_PATH FILENAME NABLA_DLL)

set(NBL_INPUT_MSVC_MANIFEST_FILE ${NBL_ROOT_PATH}/cmake/manifest/msvc/devshgraphicsprogramming.nabla.manifest)

message(STATUS "Generating ${NBL_GEN_DIRECTORY}/devshgraphicsprogramming.nabla.manifest")
configure_file("${NBL_INPUT_MSVC_MANIFEST_FILE}" "${NBL_GEN_DIRECTORY}")