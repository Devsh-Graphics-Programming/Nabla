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

# Look for the header file.
find_path(OpenCL_INCLUDE_DIR NAMES cl.h)

# Look for the library.
find_library(OpenCL_LIBRARY NAMES OpenCL)

# Handle the QUIETLY and REQUIRED arguments and set PCRE_FOUND to TRUE if all listed variables are TRUE.
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenCL DEFAULT_MSG OpenCL_LIBRARY OpenCL_INCLUDE_DIR)

# Copy the results to the output variables.
if(OpenCL_FOUND)
	SET(OpenCL_LIBRARIES ${OpenCL_LIBRARY})
	SET(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIR})
else(OpenCL_FOUND)
	SET(OpenCL_LIBRARIES)
	SET(OpenCL_INCLUDE_DIRS)
endif(OpenCL_FOUND)

MARK_AS_ADVANCED(OpenCL_INCLUDE_DIR OpenCL_LIBRARY)

if(OpenCL_FOUND AND NOT TARGET OpenCL::OpenCL)
	if(OpenCL_LIBRARY MATCHES "/([^/]+)\\.framework$")
		add_library(OpenCL::OpenCL INTERFACE IMPORTED)
		set_target_properties(OpenCL::OpenCL PROPERTIES
			INTERFACE_LINK_LIBRARIES "${OpenCL_LIBRARY}")
	else()
		add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
		set_target_properties(OpenCL::OpenCL PROPERTIES
			IMPORTED_LOCATION "${OpenCL_LIBRARY}")
	endif()
	set_target_properties(OpenCL::OpenCL PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIRS}")
endif()