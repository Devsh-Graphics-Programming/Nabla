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



# The module defines:
#
# ``NMAKE_EXECUTABLE``
#	Path to 64bit nmake
# ``NMAKE_FOUND``
#	True if the nmake executable was found
#


if (CMAKE_HOST_WIN32)
	get_filename_component(
		MY_COMPILER_DIR
		${CMAKE_CXX_COMPILER} DIRECTORY
	)
	find_program(NMAKE_EXECUTABLE
		nmake.exe
		PATHS "${MY_COMPILER_DIR}/../../Hostx64/x64" "${MY_COMPILER_DIR}"
	)
	string(COMPARE EQUAL NMAKE_EXECUTABLE NMAKE_EXECUTABLE-NOTFOUND NMAKE_FOUND)
		
	unset(MY_COMPILER_DIR)
else()
	set(NMAKE_EXECUTABLE NMAKE_EXECUTABLE-NOTFOUND)
	set(NMAKE_FOUND 0)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nmake
                                  REQUIRED_VARS NMAKE_EXECUTABLE
)
