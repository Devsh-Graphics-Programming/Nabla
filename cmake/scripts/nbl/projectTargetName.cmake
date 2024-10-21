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

# CMake script for creating target name given directory the example is placed in
# after invocation EXECUTABLE_NAME variable is set to use
# requires _NBL_PROJECT_DIRECTORY_ variable to be set

if(NOT DEFINED _NBL_PROJECT_DIRECTORY_)
	message(FATAL_ERROR "_NBL_PROJECT_DIRECTORY_ must be set to execute this script!")
endif()

get_filename_component(EXECUTABLE_NAME ${_NBL_PROJECT_DIRECTORY_} NAME)
string(REGEX REPLACE "^[0-9]+\." "" EXECUTABLE_NAME ${EXECUTABLE_NAME})
string(TOLOWER ${EXECUTABLE_NAME} EXECUTABLE_NAME)
string(MAKE_C_IDENTIFIER ${EXECUTABLE_NAME} EXECUTABLE_NAME)

if(DEFINED CI)
	execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "${EXECUTABLE_NAME}") # pipe example target name to stdout
endif()