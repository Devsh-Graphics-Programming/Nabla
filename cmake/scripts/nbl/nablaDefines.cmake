# Copyright (c) 2021 DevSH Graphics Programming Sp. z O.O.
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

# CMake script for generating Nabla.h during build time

if(DEFINED NBL_ROOT_PATH)
	if(NOT EXISTS ${NBL_ROOT_PATH})
		message(FATAL_ERROR  "NBL_ROOT_PATH as '${NBL_ROOT_PATH}' is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_ROOT_PATH variable must be specified for this script!")
endif()

if(DEFINED NBL_WRAPPER_FILE)
	if(NOT EXISTS ${NBL_WRAPPER_FILE})
		message(FATAL_ERROR  "NBL_WRAPPER_FILE as '${NBL_WRAPPER_FILE}' is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_WRAPPER_FILE variable must be specified for this script!")
endif()

if(DEFINED NBL_GEN_DIRECTORY) # directory where nabla.h will be created
	if(NOT EXISTS ${NBL_GEN_DIRECTORY})
		message(FATAL_ERROR  "NBL_GEN_DIRECTORY as '${NBL_GEN_DIRECTORY}' is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_GEN_DIRECTORY variable must be specified for this script!")
endif()

if(NOT DEFINED _NABLA_DLL_NAME_)
	message(FATAL_ERROR  "_NABLA_DLL_NAME_ variable must be specified for this script!")
endif()

if(NOT DEFINED _DXC_DLL_NAME_)
	message(FATAL_ERROR  "_DXC_DLL_NAME_ variable must be specified for this script!")
endif()

if(NOT DEFINED _NABLA_INSTALL_DIR_)
	message(FATAL_ERROR  "_NABLA_INSTALL_DIR_ variable must be specified for this script!")
endif()

configure_file("${NBL_ROOT_PATH}/cmake/install/nbl/sharedDefines.h.in" "${NBL_GEN_DIRECTORY}/define.h")
file(READ "${NBL_WRAPPER_FILE}" NBL_WRAPPER_CODE)
file(READ "${NBL_GEN_DIRECTORY}/define.h" NBL_WRAPPER_CODE_2)

string(APPEND NBL_NABLA_INSTALL_HEADER "${NBL_WRAPPER_CODE}${NBL_WRAPPER_CODE_2}")
file(WRITE "${NBL_GEN_DIRECTORY}/define.h" "${NBL_NABLA_INSTALL_HEADER}")