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

# CMake script for generating android manifest during build time

if(DEFINED NBL_ROOT_PATH)
	if(NOT EXISTS ${NBL_ROOT_PATH})
		message(FATAL_ERROR  "NBL_ROOT_PATH as '${NBL_ROOT_PATH}' is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_ROOT_PATH variable must be specified for this script!")
endif()

if(DEFINED NBL_CONFIGURATION)
	if(NBL_CONFIGURATION STREQUAL "Release")
		message(STATUS  "Generating AndroidManifest.xml for Release configuration!")
	elseif(NBL_CONFIGURATION STREQUAL "Debug")
		message(STATUS  "Generating AndroidManifest.xml for Debug configuration!")
	elseif(NBL_CONFIGURATION STREQUAL "RelWithDebInfo")
		message(STATUS  "Generating AndroidManifest.xml for RelWithDebInfo configuration!")
	else()
		message(FATAL_ERROR  "NBL_CONFIGURATION as ${NBL_CONFIGURATION} is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_CONFIGURATION variable must be specified for this script!")
endif()

if(DEFINED NBL_GEN_DIRECTORY)
	if(NOT EXISTS ${NBL_GEN_DIRECTORY})
		message(FATAL_ERROR  "NBL_GEN_DIRECTORY as '${NBL_GEN_DIRECTORY}' is invalid!")
	endif()
else()
	message(FATAL_ERROR  "NBL_GEN_DIRECTORY variable must be specified for this script!")
endif()

if(NOT DEFINED TARGET_ANDROID_API_LEVEL)
	message(FATAL_ERROR  "TARGET_ANDROID_API_LEVEL variable must be specified for this script!")
endif()

if(NOT DEFINED SO_NAME)
	message(FATAL_ERROR  "SO_NAME variable must be specified for this script!")
endif()

if(NOT DEFINED TARGET_NAME_IDENTIFIER)
	message(FATAL_ERROR  "TARGET_NAME_IDENTIFIER variable must be specified for this script!")
endif()

set(PACKAGE_NAME "eu.devsh.${TARGET_NAME_IDENTIFIER}")
set(APP_NAME ${TARGET_NAME_IDENTIFIER})

if(NBL_CONFIGURATION STREQUAL "Release")
	set(NATIVE_LIB_NAME ${SO_NAME})
elseif(NBL_CONFIGURATION STREQUAL "Debug")
	set(NATIVE_LIB_NAME ${SO_NAME}_d)
elseif(NBL_CONFIGURATION STREQUAL "RelWithDebInfo")
	set(NATIVE_LIB_NAME ${SO_NAME}_rwdi)
endif()

set(NBL_INPUT_ANDROID_MANIFEST_FILE ${NBL_ROOT_PATH}/android/AndroidManifest.xml)
set(NBL_OUTPUT_ANDROID_MANIFEST_FILE ${NBL_GEN_DIRECTORY}/${NBL_CONFIGURATION}/AndroidManifest.xml)

#configure_file(${NBL_ROOT_PATH}/android/Loader.java ${NBL_GEN_DIRECTORY}/src/eu/devsh/${NATIVE_LIB_NAME}/Loader.java)
configure_file("${NBL_INPUT_ANDROID_MANIFEST_FILE}" "${NBL_OUTPUT_ANDROID_MANIFEST_FILE}")