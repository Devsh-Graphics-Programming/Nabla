# Copyright (c) 2023 DevSH Graphics Programming Sp. z O.O.
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

if(WIN32) # TODO: to extend in future
	if(NBL_STATIC_BUILD)
		set(CPACK_PACKAGE_INSTALL_DIRECTORY "nabla-static")
		set(CPACK_PACKAGE_NAME "nabla-win64-static")
	else()
		set(CPACK_PACKAGE_INSTALL_DIRECTORY "nabla-dynamic")
		set(CPACK_PACKAGE_NAME "nabla-win64-dynamic")
	endif()

	set(CPACK_PACKAGE_VENDOR "DevshGraphicsProgramming.org")
	set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Nabla")
	set(CPACK_PACKAGE_VERSION_MAJOR "1")
	set(CPACK_PACKAGE_VERSION_MINOR "0")
	set(CPACK_PACKAGE_VERSION_PATCH "0")
	set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")

	set(CPACK_COMPONENTS_ALL Headers Libraries Runtimes Media) # TODO: add Executables conditionally

	set(CPACK_COMPONENT_HEADERS_DISPLAY_NAME "Headers")
	set(CPACK_COMPONENT_LIBRARIES_DISPLAY_NAME "Libraries")
	set(CPACK_COMPONENT_RUNTIMES_DISPLAY_NAME "Runtimes")
	#set(CPACK_COMPONENT_EXECUTABLES_DISPLAY_NAME "Examples")
	set(CPACK_COMPONENT_MEDIA_DISPLAY_NAME "Media")

	set(CPACK_COMPONENT_HEADERS_DESCRIPTION "C/C++ headers, shaders and embeded builtin resource files for use with Nabla library and extensions")
	set(CPACK_COMPONENT_LIBRARIES_DESCRIPTION "Static, import and shared libraries used to build programs with Nabla library and extensions")

	if(NBL_STATIC_BUILD)
		set(CPACK_COMPONENT_RUNTIMES_DESCRIPTION "DLL and PDB files for use with Nabla library and extensions")
	else()
		set(CPACK_COMPONENT_RUNTIMES_DESCRIPTION "PDB files for use with Nabla library and extensions")
	endif()

	#set(CPACK_COMPONENT_EXECUTABLES_DESCRIPTION "Example executables built with Nabla library")
	set(CPACK_COMPONENT_MEDIA_DESCRIPTION "Media files Nabla example executables load resources from")

	set(CPACK_COMPONENT_HEADERS_DEPENDS Libraries)
	#set(CPACK_COMPONENT_EXECUTABLES_DEPENDS Media)

	set(CPACK_COMPONENT_LIBRARIES_GROUP "Development")
	set(CPACK_COMPONENT_HEADERS_GROUP "Development")
	#set(CPACK_COMPONENT_EXECUTABLES_GROUP "Executables")
	set(CPACK_COMPONENT_MEDIA_GROUP "Executables")

	set(CPACK_THREADS 0) # try to use all threads for compression 

	include(CPack)
endif()