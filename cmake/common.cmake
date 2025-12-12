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
# limitations under the License

include_guard(GLOBAL)

include(ProcessorCount)

# Macro creating project for an executable
# Project and target get its name from directory when this macro gets executed (truncating number in the beginning of the name and making all lower case)
# Created because of common cmake code for examples and tools
macro(nbl_create_executable_project _EXTRA_SOURCES _EXTRA_OPTIONS _EXTRA_INCLUDES _EXTRA_LIBS)
	get_filename_component(_NBL_PROJECT_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
	get_filename_component(EXECUTABLE_NAME ${_NBL_PROJECT_DIRECTORY_} NAME)
	string(TOLOWER ${EXECUTABLE_NAME} EXECUTABLE_NAME)
	
	project(${EXECUTABLE_NAME})
	set_directory_properties(PROPERTIES VS_STARTUP_PROJECT ${EXECUTABLE_NAME})

	set(NBL_EXECUTABLE_SOURCES
		main.cpp
		${_EXTRA_SOURCES}
	)

	if(ANDROID)
		add_library(${EXECUTABLE_NAME} SHARED ${NBL_EXECUTABLE_SOURCES})
	else()
		add_executable(${EXECUTABLE_NAME} ${NBL_EXECUTABLE_SOURCES})
	endif()
	
	target_compile_definitions(${EXECUTABLE_NAME} PUBLIC _NBL_APP_NAME_="${EXECUTABLE_NAME}")
		
	target_include_directories(${EXECUTABLE_NAME}
		PUBLIC "${NBL_ROOT_PATH}/examples_tests/common"
		PRIVATE ${_EXTRA_INCLUDES}
	)
	target_link_libraries(${EXECUTABLE_NAME} PUBLIC Nabla ${_EXTRA_LIBS})

	nbl_adjust_flags(TARGET ${EXECUTABLE_NAME} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)	
	nbl_adjust_definitions()

	add_compile_options(${_EXTRA_OPTIONS})
	add_definitions(-D_NBL_PCH_IGNORE_PRIVATE_HEADERS) # TODO: wipe when we finally make Nabla PCH work as its supposed to
	set_target_properties(${EXECUTABLE_NAME} PROPERTIES
		DEBUG_POSTFIX _d
		RELWITHDEBINFO_POSTFIX _rwdi
		RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
		RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_SOURCE_DIR}/bin"
		RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
		VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
	)
	if(MSVC)
		# nothing special
	else() # only set up for visual studio code
		set(VSCODE_LAUNCH_JSON "
{
    \"version\": \"0.2.0\",
    \"configurations\": [
        {
            \"name\": \"(gdb) Launch\",
            \"type\": \"cppdbg\",
            \"request\": \"launch\",
            \"program\": \"${PROJECT_SOURCE_DIR}/bin/${EXECUTABLE_NAME}\",
            \"args\": [],
            \"stopAtEntry\": false,
            \"cwd\": \"${PROJECT_SOURCE_DIR}/bin\",
            \"environment\": [],
            \"externalConsole\": false,
            \"MIMode\": \"gdb\",
            \"setupCommands\": [
                {
                    \"description\": \"Enable pretty-printing for gdb\",
                    \"text\": \"-enable-pretty-printing\",
                    \"ignoreFailures\": true
                }
            ],
            \"preLaunchTask\": \"build\" 
        }
    ]
}")
		file(WRITE "${PROJECT_BINARY_DIR}/.vscode/launch.json" ${VSCODE_LAUNCH_JSON})

		ProcessorCount(CPU_COUNT)
		set(VSCODE_TASKS_JSON "
{
    \"version\": \"0.2.0\",
    \"command\": \"\",
    \"args\": [],
    \"tasks\": [
        {
            \"label\": \"build\",
            \"command\": \"${CMAKE_MAKE_PROGRAM}\",
            \"type\": \"shell\",
            \"args\": [
                \"${EXECUTABLE_NAME}\",
                \"-j${CPU_COUNT}\"
            ],
            \"options\": {
                \"cwd\": \"${CMAKE_BINARY_DIR}\"
            },
            \"group\": {
                \"kind\": \"build\",
                \"isDefault\": true
            },
            \"presentation\": {
                \"echo\": true,
                \"reveal\": \"always\",
                \"focus\": false,
                \"panel\": \"shared\"
            },
            \"problemMatcher\": \"$msCompile\"
        }
    ]
}")
		file(WRITE "${PROJECT_BINARY_DIR}/.vscode/tasks.json" ${VSCODE_TASKS_JSON})
	endif()
	if(NBL_BUILD_ANDROID)
		# https://github.com/android-ndk/ndk/issues/381
		target_link_options(${EXECUTABLE_NAME} PRIVATE -u ANativeActivity_onCreate)
		set (variadic_args ${ARGN})
		list(LENGTH variadic_args variadic_count)
		if (${variadic_count} GREATER 0)
			list(GET variadic_args 0 optional_arg)
			set(ASSET_SOURCE_DIR ${optional_arg})
			#message(FATAL_ERROR  "the path ${optional_arg} doesn't exist")     
			nbl_android_create_apk(${EXECUTABLE_NAME} ${ASSET_SOURCE_DIR})
		else()
			nbl_android_create_apk(${EXECUTABLE_NAME})
		endif ()
		
	endif()
	
	get_target_property(_EX_SOURCE_DIR_ ${EXECUTABLE_NAME} SOURCE_DIR)
	file(RELATIVE_PATH _REL_DIR_ "${NBL_ROOT_PATH}" "${_EX_SOURCE_DIR_}")
	
	if(NOT "${EXECUTABLE_NAME}" STREQUAL commonpch)
		nbl_install_exe_spec(${EXECUTABLE_NAME} "${_REL_DIR_}/bin")
		
		get_target_property(_NBL_${EXECUTABLE_NAME}_PACKAGE_RUNTIME_EXE_DIR_PATH_ ${EXECUTABLE_NAME} NBL_PACKAGE_RUNTIME_EXE_DIR_PATH)
		get_target_property(_NBL_NABLA_PACKAGE_RUNTIME_DLL_DIR_PATH_ Nabla NBL_PACKAGE_RUNTIME_DLL_DIR_PATH)
		get_property(_NBL_DXC_PACKAGE_RUNTIME_DLL_DIR_PATH_ GLOBAL PROPERTY NBL_3RDPARTY_DXC_NS_PACKAGE_RUNTIME_DLL_DIR_PATH)
		
		cmake_path(RELATIVE_PATH _NBL_NABLA_PACKAGE_RUNTIME_DLL_DIR_PATH_ BASE_DIRECTORY "${_NBL_${EXECUTABLE_NAME}_PACKAGE_RUNTIME_EXE_DIR_PATH_}" OUTPUT_VARIABLE _NBL_NABLA_PACKAGE_RUNTIME_DLL_DIR_PATH_REL_TO_TARGET_)
		cmake_path(RELATIVE_PATH _NBL_DXC_PACKAGE_RUNTIME_DLL_DIR_PATH_ BASE_DIRECTORY "${_NBL_${EXECUTABLE_NAME}_PACKAGE_RUNTIME_EXE_DIR_PATH_}" OUTPUT_VARIABLE _NBL_DXC_PACKAGE_RUNTIME_DLL_DIR_PATH_REL_TO_TARGET_)
		
		# DLLs relative to CPack install package
		target_compile_definitions(${EXECUTABLE_NAME}
			PRIVATE "-DNBL_CPACK_PACKAGE_NABLA_DLL_DIR=\"${_NBL_NABLA_PACKAGE_RUNTIME_DLL_DIR_PATH_REL_TO_TARGET_}\"" 
			PRIVATE	"-DNBL_CPACK_PACKAGE_DXC_DLL_DIR=\"${_NBL_DXC_PACKAGE_RUNTIME_DLL_DIR_PATH_REL_TO_TARGET_}\""
		)
	endif()

	nbl_project_process_test_module()
endmacro()

macro(nbl_create_ext_library_project EXT_NAME LIB_HEADERS LIB_SOURCES LIB_INCLUDES LIB_OPTIONS DEF_OPTIONS)
	set(LIB_NAME "NblExt${EXT_NAME}")
	project(${LIB_NAME})

	add_library(${LIB_NAME} ${LIB_SOURCES})
	
	target_link_libraries(${LIB_NAME} PUBLIC Nabla)
	target_include_directories(${LIB_NAME} PRIVATE ${LIB_INCLUDES})

	nbl_adjust_flags(TARGET ${LIB_NAME} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
	nbl_adjust_definitions()

	target_compile_options(${LIB_NAME} PUBLIC ${LIB_OPTIONS})
	target_compile_definitions(${LIB_NAME} PUBLIC ${DEF_OPTIONS})
	set_target_properties(${LIB_NAME} PROPERTIES
		DEBUG_POSTFIX _d
		RELWITHDEBINFO_POSTFIX _rwdi
	)
	
	if(LIB_HEADERS)
		nbl_install_file_spec(${LIB_HEADERS} "nbl/ext/${EXT_NAME}")
	endif()
	
	nbl_install_lib_spec(${LIB_NAME} "nbl/ext/${EXT_NAME}")
	
	set("NBL_EXT_${EXT_NAME}_INCLUDE_DIRS"
		"${NBL_ROOT_PATH}/include/"
		"${NBL_ROOT_PATH}/src"
		"${NBL_ROOT_PATH}/source/Nabla"
		"${NBL_ROOT_PATH}/ext/${EXT_NAME}"
		"${LIB_INCLUDES}"
		PARENT_SCOPE
	)
	set("NBL_EXT_${EXT_NAME}_LIB"
		"${LIB_NAME}"
		PARENT_SCOPE
	)
endmacro()

###########################################
# Nabla install rules, directory structure:
#
# -	include 				(portable header files)
# - $<CONFIG>/lib 			(static or import shared libraries)
# - $<CONFIG>/runtime 		(DLLs/SOs/PDBs)
# - $<CONFIG>/exe			(executables and media)
#
# If $<CONFIG> == Release, then the directory structure doesn't begin with $<CONFIG>

set(_NBL_CPACK_PACKAGE_RELATIVE_ENTRY_ "./$<$<NOT:$<STREQUAL:$<CONFIG>,Release>>:$<LOWER_CASE:$<CONFIG>>>")

function(nbl_install_headers_spec _HEADERS _BASE_HEADERS_DIR)
	foreach (file ${_HEADERS})
		file(RELATIVE_PATH dir ${_BASE_HEADERS_DIR} ${file})
		get_filename_component(dir ${dir} DIRECTORY)
		install(FILES ${file} DESTINATION include/${dir} COMPONENT Headers)
	endforeach()
endfunction()

function(nbl_install_headers _HEADERS)
	if(NOT DEFINED NBL_ROOT_PATH)
		message(FATAL_ERROR "NBL_ROOT_PATH isn't defined!")
	endif()

	nbl_install_headers_spec("${_HEADERS}" "${NBL_ROOT_PATH}/include")
endfunction()

function(nbl_install_file_spec _FILES _RELATIVE_DESTINATION)
	install(FILES ${_FILES} DESTINATION include/${_RELATIVE_DESTINATION} COMPONENT Headers)
endfunction()

function(nbl_install_file _FILES)
	nbl_install_file_spec("${_FILES}" "")
endfunction()

function(nbl_install_dir_spec _DIR _RELATIVE_DESTINATION)
	install(DIRECTORY ${_DIR} DESTINATION include/${_RELATIVE_DESTINATION} COMPONENT Headers)
endfunction()

function(nbl_install_dir _DIR)
	nbl_install_dir_spec("${_DIR}" "")
endfunction()

function(nbl_install_lib_spec _TARGETS _RELATIVE_DESTINATION)
	install(TARGETS ${_TARGETS} ARCHIVE DESTINATION lib/${_RELATIVE_DESTINATION} CONFIGURATIONS Release COMPONENT Libraries)
	install(TARGETS ${_TARGETS} ARCHIVE DESTINATION debug/lib/${_RELATIVE_DESTINATION} CONFIGURATIONS Debug COMPONENT Libraries)
	install(TARGETS ${_TARGETS} ARCHIVE DESTINATION relwithdebinfo/lib/${_RELATIVE_DESTINATION} CONFIGURATIONS RelWithDebInfo COMPONENT Libraries)
endfunction()

function(nbl_install_lib _TARGETS)
	nbl_install_lib_spec("${_TARGETS}" "")
endfunction()

function(nbl_install_program_spec _TRGT _RELATIVE_DESTINATION)
	set(_DEST_GE_ "${_NBL_CPACK_PACKAGE_RELATIVE_ENTRY_}/runtime/${_RELATIVE_DESTINATION}")
	
	if (TARGET ${_TRGT})
		foreach(_CONFIGURATION_ IN LISTS CMAKE_CONFIGURATION_TYPES)
			install(PROGRAMS $<TARGET_FILE:${_TRGT}> DESTINATION ${_DEST_GE_} CONFIGURATIONS ${_CONFIGURATION_} COMPONENT Runtimes)
		endforeach()
	
		install(PROGRAMS $<TARGET_PDB_FILE:${_TRGT}> DESTINATION debug/runtime/${_RELATIVE_DESTINATION} CONFIGURATIONS Debug COMPONENT Runtimes) # TODO: write cmake script with GE to detect if target in configuration has PDB files generated then add install rule
		
		get_property(_DEFINED_PROPERTY_
            TARGET ${_TRGT}
            PROPERTY NBL_PACKAGE_RUNTIME_DLL_DIR_PATH
            DEFINED
		)
		
		if(NOT _DEFINED_PROPERTY_)
			define_property(TARGET
                PROPERTY NBL_PACKAGE_RUNTIME_DLL_DIR_PATH
                BRIEF_DOCS "Relative path in CPack package to runtime DLL directory"
			)
		endif()
		
		set_target_properties(${_TRGT} PROPERTIES NBL_PACKAGE_RUNTIME_DLL_DIR_PATH "${_DEST_GE_}")
		
	else()
		foreach(_CONFIGURATION_ IN LISTS CMAKE_CONFIGURATION_TYPES)
			install(PROGRAMS ${_TRGT} DESTINATION ${_DEST_GE_} CONFIGURATIONS ${_CONFIGURATION_} COMPONENT Runtimes)
		endforeach()
		
		string(MAKE_C_IDENTIFIER "${_RELATIVE_DESTINATION}" _VAR_)
		string(TOUPPER "${_VAR_}" _VAR_)
		
		get_property(_DEFINED_PROPERTY_
            GLOBAL
            PROPERTY ${_VAR_}_NS_PACKAGE_RUNTIME_DLL_DIR_PATH
            DEFINED
		)
		
		if(NOT _DEFINED_PROPERTY_)
			define_property(GLOBAL
                PROPERTY ${_VAR_}_NS_PACKAGE_RUNTIME_DLL_DIR_PATH
                BRIEF_DOCS "Relative path in CPack package to runtime DLL directory"
			)
		endif()
		
		set_property(GLOBAL PROPERTY ${_VAR_}_NS_PACKAGE_RUNTIME_DLL_DIR_PATH "${_DEST_GE_}")
	endif()
endfunction()

function(nbl_install_program _TRGT)
	nbl_install_program_spec("${_TRGT}" "")
endfunction()

function(nbl_install_exe_spec _TARGETS _RELATIVE_DESTINATION)
	set(_TARGETS ${_TARGETS})
	set(_DEST_GE_ "${_NBL_CPACK_PACKAGE_RELATIVE_ENTRY_}/exe/${_RELATIVE_DESTINATION}")
	
	foreach(_CONFIGURATION_ IN LISTS CMAKE_CONFIGURATION_TYPES)
		install(TARGETS ${_TARGETS} RUNTIME DESTINATION ${_DEST_GE_} CONFIGURATIONS ${_CONFIGURATION_} COMPONENT Executables)
	endforeach()
	
	foreach(_TRGT IN LISTS _TARGETS)
		get_property(_DEFINED_PROPERTY_
			TARGET ${_TRGT}
			PROPERTY NBL_PACKAGE_RUNTIME_EXE_DIR_PATH
			DEFINED
		)
		
		if(NOT _DEFINED_PROPERTY_)
			define_property(TARGET
				PROPERTY NBL_PACKAGE_RUNTIME_EXE_DIR_PATH
				BRIEF_DOCS "Relative path in CPack package to runtime executable target directory"
			)
		endif()
		
		set_target_properties(${_TRGT} PROPERTIES NBL_PACKAGE_RUNTIME_EXE_DIR_PATH "${_DEST_GE_}")
	endforeach()
endfunction()

function(nbl_install_exe _TARGETS)
	nbl_install_exe_spec("${_TARGETS}" "")
endfunction()

function(nbl_install_media_spec _FILE _RELATIVE_DESTINATION)
	install(FILES ${_FILE} DESTINATION exe/${_RELATIVE_DESTINATION} CONFIGURATIONS Release COMPONENT Media EXCLUDE_FROM_ALL)
	install(FILES ${_FILE} DESTINATION debug/exe/${_RELATIVE_DESTINATION} CONFIGURATIONS Debug COMPONENT Media EXCLUDE_FROM_ALL)
	install(FILES ${_FILE} DESTINATION relwithdebinfo/exe/${_RELATIVE_DESTINATION} CONFIGURATIONS RelWithDebInfo COMPONENT Media EXCLUDE_FROM_ALL)
endfunction()

function(nbl_install_media _FILE)
	nbl_install_lib_spec("${_FILE}" "")
endfunction()

function(nbl_install_builtin_resources _TARGET_)
	get_target_property(_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_ ${_TARGET_} BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
	get_target_property(_BUILTIN_RESOURCES_HEADERS_ ${_TARGET_} BUILTIN_RESOURCES_HEADERS)
	
	nbl_install_headers_spec("${_BUILTIN_RESOURCES_HEADERS_}" "${_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_}")
endfunction()

function(NBL_TEST_MODULE_INSTALL_FILE _NBL_FILEPATH_)
	file(RELATIVE_PATH _NBL_REL_INSTALL_DEST_ "${NBL_ROOT_PATH}" "${_NBL_FILEPATH_}")
	cmake_path(GET _NBL_REL_INSTALL_DEST_ PARENT_PATH _NBL_REL_INSTALL_DEST_)
					
	nbl_install_media_spec("${_NBL_FILEPATH_}" "${_NBL_REL_INSTALL_DEST_}")
endfunction()

function(nbl_project_process_test_module)
	set(NBL_TEMPLATE_JSON_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
	get_filename_component(NBL_TEMPLATE_JSON_DIR_ABS_ "${NBL_TEMPLATE_JSON_DIR}" ABSOLUTE)
	set(NBL_PROFILES_JSON_DIR "${NBL_TEMPLATE_JSON_DIR}/.profiles")
	get_filename_component(NBL_PROFILES_JSON_DIR_ABS_ "${NBL_PROFILES_JSON_DIR}" ABSOLUTE)

 	file(RELATIVE_PATH NBL_ROOT_PATH_REL "${NBL_TEMPLATE_JSON_DIR}" "${NBL_ROOT_PATH}")
	cmake_path(GET NBL_TEMPLATE_JSON_DIR FILENAME _NBL_PF_MODULE_NAME_)
	set(_NBL_PF_DESC_NAME_ "${_NBL_PF_MODULE_NAME_} test module")

	set(NBL_EXECUTABLE_GEN_EXP_FILEPATH "$<PATH:RELATIVE_PATH,$<TARGET_FILE:${EXECUTABLE_NAME}>,${NBL_PROFILES_JSON_DIR}>") # use this in your json config file when referencing filepath of an example's executable

	set(_NBL_JSON_CONFIG_FILEPATH_ "${NBL_TEMPLATE_JSON_DIR}/config.json.template")

	macro(NBL_JSON_READ_VALIDATE_POPULATE _NBL_FIELD_TRAVERSAL_ _NBL_FIELD_NAME_ _NBL_EXPECTED_TYPE_ _NBL_JSON_CONTENT_)
		string(TOUPPER "${_NBL_FIELD_NAME_}" _NBL_FIELD_NAME_UPPER_)
		
		string(JSON _NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_CONTENT_ ERROR_VARIABLE _NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_CONTENT_ERR_ GET ${_NBL_JSON_CONTENT_} ${_NBL_FIELD_NAME_})
		string(JSON _NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_LEN_ ERROR_VARIABLE _NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_LEN_ERR_ LENGTH ${_NBL_JSON_CONTENT_} ${_NBL_FIELD_NAME_})
		string(JSON _NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_TYPE_ ERROR_VARIABLE _NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_TYPE_ERR_ TYPE ${_NBL_JSON_CONTENT_} ${_NBL_FIELD_NAME_})
		
		if(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION)
			set(_JSON_TYPE_ "${_NBL_JSON_${_NBL_FIELD_NAME_UPPER_}_TYPE_}")
			if(NOT "${_JSON_TYPE_}" STREQUAL "${_NBL_EXPECTED_TYPE_}") # validate type
				message(FATAL_ERROR "\"${_NBL_JSON_CONFIG_FILEPATH_}\" validation failed! \"${_NBL_FIELD_TRAVERSAL_}.${_NBL_FIELD_NAME_}\" field is not \"${_NBL_EXPECTED_TYPE_}\" type but \"${_JSON_TYPE_}\".")
			endif()
			unset(_JSON_TYPE_)
		endif()
		
		unset(_NBL_FIELD_NAME_UPPER_)
	endmacro()
	
	macro(NBL_READ_VALIDATE_INSTALL_JSON_DEPENDENCIES_SPEC _NBL_INPUT_JSON_CONFIG_FILEPATH_ _NBL_FIELD_TRAVERSAL_INCLUSIVE_ _NBL_JSON_DEPENDENCIES_LIST_CONTENT_)
		set(_NBL_INPUT_JSON_CONFIG_FILEPATH_ "${_NBL_INPUT_JSON_CONFIG_FILEPATH_}")
		cmake_path(GET _NBL_INPUT_JSON_CONFIG_FILEPATH_ PARENT_PATH _NBL_INPUT_JSON_LOCATION_)
		
		string(JSON _NBL_JSON_DEPENDENCIES_LIST_CONTENT_LEN_ ERROR_VARIABLE _NBL_JSON_DEPENDENCIES_LIST_CONTENT_LEN_ERR_ LENGTH ${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_})
		string(JSON _NBL_JSON_DEPENDENCIES_LIST_CONTENT_TYPE_ ERROR_VARIABLE _NBL_JSON_DEPENDENCIES_LIST_CONTENT_TYPE_ERR_ TYPE ${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_})
	
		if(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION)
			if(NOT "${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_TYPE_}" STREQUAL ARRAY) # validate type
				message(FATAL_ERROR "Internal error while processing \"${_NBL_INPUT_JSON_CONFIG_FILEPATH_}\". Wrong usage of NBL_READ_VALIDATE_INSTALL_JSON_DEPENDENCIES macro, input _NBL_JSON_DEPENDENCIES_LIST_CONTENT_ = \"${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_}\" is not ARRAY type!")
			endif()
		endif()
	
		if(_NBL_JSON_DEPENDENCIES_LIST_CONTENT_LEN_ GREATER_EQUAL 1)
			math(EXPR _NBL_STOP_ "${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_LEN_}-1")
			
			foreach(_NBL_IDX_ RANGE ${_NBL_STOP_})
				string(JSON _NBL_JSON_DEPENDENCIES_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_} ${_NBL_IDX_})
				
				set(_NBL_JSON_DEPENDENCY_FILEPATH_ "${_NBL_INPUT_JSON_LOCATION_}/${_NBL_JSON_DEPENDENCIES_LIST_ELEMENT_CONTENT_}") # json config file may reference files relative to itself
				get_filename_component(_NBL_JSON_DEPENDENCY_FILEPATH_ABS_ "${_NBL_JSON_DEPENDENCY_FILEPATH_}" ABSOLUTE)
				
				if(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION)
					if(NOT EXISTS "${_NBL_JSON_DEPENDENCY_FILEPATH_}") # validate if present
						message(FATAL_ERROR "Declared \"${_NBL_INPUT_JSON_CONFIG_FILEPATH_}\"'s ${_NBL_FIELD_TRAVERSAL_INCLUSIVE_}[${_NBL_IDX_}] = \"${_NBL_JSON_DEPENDENCIES_LIST_ELEMENT_CONTENT_}\" doesn't exist! It's filepath is resolved to \"${_NBL_JSON_DEPENDENCY_FILEPATH_ABS_}\". Note that filepaths in json configs are resolved relative to them.")
					endif()
				endif()
				
				string(FIND "${_NBL_JSON_DEPENDENCY_FILEPATH_ABS_}" "${NBL_MEDIA_DIRECTORY_ABS}" _NBL_IS_MEDIA_DEPENDNECY_)
				
				if(NOT "${_NBL_IS_MEDIA_DEPENDNECY_}" STREQUAL "-1") # filter dependencies, only those coming from NBL_MEDIA_DIRECTORY_ABS are considered for install rules
					NBL_TEST_MODULE_INSTALL_FILE("${_NBL_JSON_DEPENDENCY_FILEPATH_}")
				endif()
			endforeach()
		endif()
	endmacro()
	
	macro(NBL_READ_VALIDATE_INSTALL_JSON_DEPENDENCIES _NBL_FIELD_TRAVERSAL_INCLUSIVE_ _NBL_JSON_DEPENDENCIES_LIST_CONTENT_)
		NBL_READ_VALIDATE_INSTALL_JSON_DEPENDENCIES_SPEC("${_NBL_JSON_CONFIG_FILEPATH_}" "${_NBL_FIELD_TRAVERSAL_INCLUSIVE_}" "${_NBL_JSON_DEPENDENCIES_LIST_CONTENT_}")
	endmacro()
	
	# adjust relative path of any non generator expression resource to generated profile directory
	# use it after validation of the requested field
	# it expects _NBL_INPUT_JSON_CONFIG_FILEPATH_ _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ <...>
	# where _NBL_INPUT_JSON_CONFIG_FILEPATH_ is an input JSON configuration file for which part of its content may be referenced and passed in a fold expression
	#_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ is name of a CMake variable JSON result will be stored in
	# and <...> is a fold expression with input arguments for CMake string JSON GET https://cmake.org/cmake/help/latest/command/string.html#json-get beginning with <json-string>
	# if used differently it will considered undefined behaviour
	# supports referencing string and array of strings
	
	function(NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH _NBL_INPUT_JSON_CONFIG_FILEPATH_ _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_)
		cmake_path(GET _NBL_INPUT_JSON_CONFIG_FILEPATH_ PARENT_PATH _NBL_INPUT_JSON_CONFIG_DIR_)
	
		# copy fold expression arguments
		list(SUBLIST ARGV 2 ${ARGC} _ARGV_F_COPY_)
		list(LENGTH _ARGV_F_COPY_ _ARGC_F_COPY_)
		
		macro(NBL_IMPL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH _NBL_IMPL_JSON_READ_REFERENCE_FILEPATH_CONTENT_)
			set(_NBL_JSON_REFERENCE_FILEPATH_ABS_ "${_NBL_INPUT_JSON_CONFIG_DIR_}/${_NBL_IMPL_JSON_READ_REFERENCE_FILEPATH_CONTENT_}")

			if(EXISTS "${_NBL_JSON_REFERENCE_FILEPATH_ABS_}")
				file(RELATIVE_PATH _NBL_JSON_GEN_REFERENCE_FILEPATH_ "${NBL_PROFILES_JSON_DIR_ABS_}" "${_NBL_JSON_REFERENCE_FILEPATH_ABS_}")

				list(SUBLIST _ARGV_F_COPY_ 1 ${_ARGC_F_COPY_} _NBL_MEMBER_INDEX_ARGV_)
			
				if(${ARGC} GREATER_EQUAL 2)
					list(APPEND _NBL_MEMBER_INDEX_ARGV_ "${ARGV1}")
				endif()

				string(JSON ${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_} ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" ${_NBL_MEMBER_INDEX_ARGV_} "\"${_NBL_JSON_GEN_REFERENCE_FILEPATH_}\"")
				set(${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_} "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" PARENT_SCOPE)
			endif()
		endmacro()
	
		string(JSON _NBL_JSON_READ_REFERENCE_FILEPATH_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_ARGV_F_COPY_})
		string(JSON _NBL_JSON_READ_REFERENCE_FILEPATH_TYPE_ ERROR_VARIABLE _NBL_JSON_ERROR_ TYPE ${_ARGV_F_COPY_})
		
		if("${_NBL_JSON_READ_REFERENCE_FILEPATH_TYPE_}" STREQUAL STRING)
			NBL_IMPL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH("${_NBL_JSON_READ_REFERENCE_FILEPATH_CONTENT_}")
		elseif("${_NBL_JSON_READ_REFERENCE_FILEPATH_TYPE_}" STREQUAL ARRAY)
			string(JSON _NBL_JSON_READ_REFERENCE_FILEPATH_LEN_ ERROR_VARIABLE _NBL_JSON_ERROR_ LENGTH ${_ARGV_F_COPY_})
		
			if(_NBL_JSON_READ_REFERENCE_FILEPATH_LEN_ GREATER_EQUAL 1)
				math(EXPR _NBL_STOP_ "${_NBL_JSON_READ_REFERENCE_FILEPATH_LEN_}-1")
				
				foreach(_NBL_IDX_ RANGE ${_NBL_STOP_})
					string(JSON _NBL_JSON_READ_REFERENCE_FILEPATH_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_ARGV_F_COPY_} ${_NBL_IDX_})
					string(JSON _NBL_JSON_READ_REFERENCE_FILEPATH_TYPE_ ERROR_VARIABLE _NBL_JSON_ERROR_ TYPE ${_ARGV_F_COPY_} ${_NBL_IDX_})
					
					if(NOT "${_NBL_JSON_READ_REFERENCE_FILEPATH_TYPE_}" STREQUAL STRING)
						message(FATAL_ERROR "Internal error while processing \"${_NBL_JSON_CONFIG_FILEPATH_}\". NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH(${ARGV}) invocation failed while processing, referenced json array object contains an element that is not a string type!")
					endif()
					
					NBL_IMPL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH("${_NBL_JSON_READ_REFERENCE_FILEPATH_CONTENT_}" ${_NBL_IDX_})
				endforeach()
			endif()
		else()
			message(FATAL_ERROR "Internal error while processing \"${_NBL_JSON_CONFIG_FILEPATH_}\". NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH(${ARGV}) invocation failed while processing, referenced json object isn't string neither array type!")
		endif()
	endfunction()
	
	if(EXISTS "${_NBL_JSON_CONFIG_FILEPATH_}")		
		file(READ "${_NBL_JSON_CONFIG_FILEPATH_}" _NBL_JSON_TOP_CONFIG_CONTENT_)
		set(_NBL_GEN_JSON_TOP_CONFIG_CONTENT_ "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
		
		target_sources(${EXECUTABLE_NAME} PUBLIC "${_NBL_JSON_CONFIG_FILEPATH_}")
		source_group("${NBL_PFT_SOURCE_GROUP_NAME}/target/JSON" FILES "${_NBL_JSON_CONFIG_FILEPATH_}")

		# ".scriptPath" string
		NBL_JSON_READ_VALIDATE_POPULATE("" scriptPath STRING "${_NBL_JSON_TOP_CONFIG_CONTENT_}")

		if("${_NBL_JSON_SCRIPTPATH_CONTENT_}" STREQUAL "") # stop processing if script path field is empty
			return()
		endif()
		
		get_filename_component(_NBL_JSON_SCRIPTPATH_ABS_ "${NBL_TEMPLATE_JSON_DIR}/${_NBL_JSON_SCRIPTPATH_CONTENT_}" ABSOLUTE)
		if(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION)
			if(EXISTS "${_NBL_JSON_SCRIPTPATH_ABS_}")
				string(FIND "${_NBL_JSON_SCRIPTPATH_ABS_}" "${NBL_TEMPLATE_JSON_DIR_ABS_}" _NBL_FOUND_)
				
				if("${_NBL_FOUND_}" STREQUAL "-1") # always fail validation regardless .isExecuted field if the script exist and is outside testing environment directory
					message(FATAL_ERROR "\"${_NBL_JSON_CONFIG_FILEPATH_}\" validation failed! \".scriptPath\" = \"${_NBL_JSON_SCRIPTPATH_CONTENT_}\" is located outside testing environment, it must be moved anywhere to \"${NBL_TEMPLATE_JSON_DIR_ABS_}/*\" location. It's filepath is resolved to \"${_NBL_JSON_SCRIPTPATH_ABS_}\". Note that filepaths in json configs are resolved relative to them.")
				endif()
			else()
				if(_NBL_JSON_ISEXECUTED_CONTENT_)
					set(_NBL_MESSAGE_STATUS_ failed)
					set(_NBL_MESSAGE_TYPE FATAL_ERROR)
				else()
					set(_NBL_MESSAGE_STATUS_ warning)
					set(_NBL_MESSAGE_TYPE WARNING)
				endif()
				
				message(${_NBL_MESSAGE_TYPE} "\"${_NBL_JSON_CONFIG_FILEPATH_}\" validation ${_NBL_MESSAGE_STATUS_}! \".isExecuted\" field is set to \"${_NBL_JSON_ISEXECUTED_CONTENT_}\" but \".scriptPath\" = \"${_NBL_JSON_SCRIPTPATH_CONTENT_}\" doesn't exist! It's filepath is resolved to \"${_NBL_JSON_SCRIPTPATH_ABS_}\". Note that filepaths in json configs are resolved relative to them.")
			endif()
		endif()

		NBL_TEST_MODULE_INSTALL_FILE("${_NBL_JSON_SCRIPTPATH_ABS_}")

		# ".enableParallelBuild" boolean
		NBL_JSON_READ_VALIDATE_POPULATE("" enableParallelBuild BOOLEAN "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
		
		# ".threadsPerBuildProcess" number
		NBL_JSON_READ_VALIDATE_POPULATE("" threadsPerBuildProcess NUMBER "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
		
		# ".isExecuted" boolean
		NBL_JSON_READ_VALIDATE_POPULATE("" isExecuted BOOLEAN "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
	
		# configure python environment & test runtime setup script
		file(RELATIVE_PATH NBL_TEST_TARGET_MODULE_PATH_REL "${NBL_TEMPLATE_JSON_DIR}" "${_NBL_JSON_SCRIPTPATH_ABS_}")
		cmake_path(GET NBL_TEST_TARGET_MODULE_PATH_REL PARENT_PATH NBL_TEST_TARGET_MODULE_PATH_REL)
		cmake_path(GET NBL_PYTHON_FRAMEWORK_RUNALLTESTS_SCRIPT_ABS STEM LAST_ONLY NBL_RUNALLTESTS_SCRIPT_FILENAME)

		file(RELATIVE_PATH NBL_PYTHON_FRAMEWORK_MODULE_PATH_REL "${NBL_TEMPLATE_JSON_DIR}" "${NBL_PYTHON_MODULE_ROOT_PATH}/src")
		cmake_path(GET _NBL_JSON_SCRIPTPATH_ABS_ STEM LAST_ONLY NBL_TEST_TARGET_INTERFACE_SCRIPT_NAME)
		
		set(NBL_RUNALLTESTS_SCRIPT_FILEPATH "${NBL_TEMPLATE_JSON_DIR}/${NBL_RUNALLTESTS_SCRIPT_FILENAME}")
		configure_file("${NBL_PYTHON_FRAMEWORK_RUNALLTESTS_SCRIPT_ABS}" "${NBL_RUNALLTESTS_SCRIPT_FILEPATH}" @ONLY)
		NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH("${_NBL_JSON_CONFIG_FILEPATH_}" _NBL_GEN_JSON_TOP_CONFIG_CONTENT_ "${_NBL_JSON_TOP_CONFIG_CONTENT_}" scriptPath)
		NBL_TEST_MODULE_INSTALL_FILE("${NBL_RUNALLTESTS_SCRIPT_FILEPATH}")

		configure_file("${NBL_PYTHON_FRAMEWORK_VS_LAUNCH_JSON_ABS}" "${NBL_TEMPLATE_JSON_DIR}/.vscode/launch.json" @ONLY)
		configure_file("${NBL_PYTHON_FRAMEWORK_VS_SETTINGS_JSON_ABS}" "${NBL_TEMPLATE_JSON_DIR}/.vscode/settings.json" @ONLY)
		
		NBL_TEST_MODULE_INSTALL_FILE("${NBL_TEMPLATE_JSON_DIR}/.vscode/launch.json")
		NBL_TEST_MODULE_INSTALL_FILE("${NBL_TEMPLATE_JSON_DIR}/.vscode/settings.json")

		target_sources(${EXECUTABLE_NAME} PUBLIC "${NBL_RUNALLTESTS_SCRIPT_FILEPATH}")
		source_group("${NBL_PFT_SOURCE_GROUP_NAME}/target" FILES "${NBL_RUNALLTESTS_SCRIPT_FILEPATH}")
		
		target_sources(${EXECUTABLE_NAME} PUBLIC
			"${_NBL_JSON_SCRIPTPATH_ABS_}"
		)
		source_group("${NBL_PFT_SOURCE_GROUP_NAME}/target/Interface" FILES 
			"${_NBL_JSON_SCRIPTPATH_ABS_}"
		)
		
		# ".cmake" object
		NBL_JSON_READ_VALIDATE_POPULATE("" cmake OBJECT "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
		
		# ".cmake.buildModes array"
		NBL_JSON_READ_VALIDATE_POPULATE("" buildModes ARRAY "${_NBL_JSON_CMAKE_CONTENT_}")
		
		# ".cmake.requiredOptions array"
		NBL_JSON_READ_VALIDATE_POPULATE("" requiredOptions ARRAY "${_NBL_JSON_CMAKE_CONTENT_}")
		
		if(_NBL_JSON_REQUIREDOPTIONS_LEN_ GREATER_EQUAL 1)
			math(EXPR _NBL_STOP_ "${_NBL_JSON_REQUIREDOPTIONS_LEN_}-1")
			
			foreach(_NBL_IDX_ RANGE ${_NBL_STOP_})
				string(JSON _NBL_JSON_CMAKE_REQUIRED_OPTIONS_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_NBL_JSON_REQUIREDOPTIONS_CONTENT_} ${_NBL_IDX_})
				
				if(NBL_ENABLE_PROJECT_JSON_CONFIG_VALIDATION)
					if(NOT DEFINED "${_NBL_JSON_CMAKE_REQUIRED_OPTIONS_LIST_ELEMENT_CONTENT_}")
						message(FATAL_ERROR "\"${_NBL_JSON_CONFIG_FILEPATH_}\" validation failed! \"${_NBL_JSON_CMAKE_REQUIRED_OPTIONS_LIST_ELEMENT_CONTENT_}\" CMake variable is required but is not defined!")
					endif()
				endif()
			endforeach()
		endif()

		# an utility to handle single json input - ".dependencies" and ".data" fields

		macro(NBL_JSON_HANDLE_INPUT_CONFIG_PART _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ _NBL_INPUT_JSON_CONFIG_FILEPATH_ _NBL_INPUT_JSON_CONFIG_CONTENT_)
			# ".dependencies" array
			NBL_JSON_READ_VALIDATE_POPULATE("" dependencies ARRAY "${_NBL_INPUT_JSON_CONFIG_CONTENT_}")
			NBL_READ_VALIDATE_INSTALL_JSON_DEPENDENCIES_SPEC("${_NBL_INPUT_JSON_CONFIG_FILEPATH_}" ".dependencies" "${_NBL_JSON_DEPENDENCIES_CONTENT_}")
			
			NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH("${_NBL_INPUT_JSON_CONFIG_FILEPATH_}" "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" "${_NBL_INPUT_JSON_CONFIG_CONTENT_}" dependencies)
			
			# ".data" array
			NBL_JSON_READ_VALIDATE_POPULATE("" data ARRAY "${_NBL_INPUT_JSON_CONFIG_CONTENT_}")
			set(_NBL_GEN_JSON_DATA_CONTENT_ "${_NBL_JSON_DATA_CONTENT_}")
			
			if(_NBL_JSON_DATA_LEN_ GREATER_EQUAL 1)
				math(EXPR _NBL_STOP_ "${_NBL_JSON_DATA_LEN_}-1")
				
				foreach(_NBL_IDX_ RANGE ${_NBL_STOP_})
					string(JSON _NBL_JSON_DATA_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_NBL_JSON_DATA_CONTENT_} ${_NBL_IDX_})
					set(_NBL_JSON_FIELD_TRAVERSAL_ "data[${_NBL_IDX_}]")
					
					# "${_NBL_JSON_FIELD_TRAVERSAL_}.command" array
					NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" command ARRAY "${_NBL_JSON_DATA_LIST_ELEMENT_CONTENT_}")
					if(_NBL_JSON_COMMAND_LEN_ GREATER_EQUAL 1)
						math(EXPR _NBL_STOP_2_ "${_NBL_JSON_COMMAND_LEN_}-1")
							
						NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH("${_NBL_INPUT_JSON_CONFIG_FILEPATH_}" "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" "${_NBL_INPUT_JSON_CONFIG_CONTENT_}" data ${_NBL_IDX_} command)

						foreach(_NBL_IDX_2_ RANGE ${_NBL_STOP_2_})
							string(JSON _NBL_JSON_COMMAND_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" data ${_NBL_IDX_} command ${_NBL_IDX_2_})
							string(APPEND _NBL_COMMAND_CONCAT_ "${_NBL_JSON_COMMAND_LIST_ELEMENT_CONTENT_} ") # parse command arguments, concatenate
						endforeach()
					endif()
					
					string(STRIP "${_NBL_COMMAND_CONCAT_}" _NBL_COMMAND_CONCAT_)
					string(JSON "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" data ${_NBL_IDX_} command "\"${_NBL_COMMAND_CONCAT_}\"") # override command field to become concatenated string
					unset(_NBL_COMMAND_CONCAT_)
					
					# "${_NBL_JSON_FIELD_TRAVERSAL_}.dependencies" array
					NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" dependencies ARRAY "${_NBL_JSON_DATA_LIST_ELEMENT_CONTENT_}")		
					NBL_READ_VALIDATE_INSTALL_JSON_DEPENDENCIES_SPEC("${_NBL_INPUT_JSON_CONFIG_FILEPATH_}" ".${_NBL_JSON_FIELD_TRAVERSAL_}.dependencies" "${_NBL_JSON_DEPENDENCIES_CONTENT_}")
					
					NBL_JSON_UPDATE_RELATIVE_REFERENCE_FILEPATH("${_NBL_INPUT_JSON_CONFIG_FILEPATH_}" "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" "${_NBL_INPUT_JSON_CONFIG_CONTENT_}" data ${_NBL_IDX_} dependencies)
				endforeach()
			endif()
		endmacro()

		# an utility to merge content from input json to create one collection of batches
		# (stored in a variable which name is passed to as an argument) containing .dependencies and .data fields 

		# _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ is name of a variable for which json output will be stored to
		# _NBL_GEN_INPUT_JSON_CONFIG_CONTENT_ is content of an input json config the content merge with be perfrom with

		macro(NBL_JSON_MERGE_INPUT_CONFIGS_CONTENT _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ _NBL_GEN_INPUT_JSON_CONFIG_CONTENT_)
			string(JSON _NBL_GEN_JSON_INPUT_CONFIG_DEPENDENCIES_LENGTH_ ERROR_VARIABLE _NBL_JSON_ERROR_ LENGTH ${_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_} dependencies)
			if(_NBL_GEN_JSON_INPUT_CONFIG_DEPENDENCIES_LENGTH_ AND _NBL_GEN_JSON_INPUT_CONFIG_DEPENDENCIES_LENGTH_ GREATER_EQUAL 1)
				string(JSON _NBL_GEN_JSON_TOP_CONFIG_DEPENDENCIES_LENGTH_ ERROR_VARIABLE _NBL_JSON_ERROR_ LENGTH "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" dependencies)
								
				math(EXPR _NBL_STOP_2_ "${_NBL_GEN_JSON_INPUT_CONFIG_DEPENDENCIES_LENGTH_}-1")

				foreach(_NBL_IDX_2_ RANGE "${_NBL_STOP_2_}")
					math(EXPR _NBL_IDX_2_WITH_OFFSET_ "${_NBL_IDX_2_}+${_NBL_GEN_JSON_TOP_CONFIG_DEPENDENCIES_LENGTH_}")
							
					string(JSON _NBL_JSON_DEPENDENCIES_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_} dependencies ${_NBL_IDX_2_})
					string(JSON "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" dependencies ${_NBL_IDX_2_WITH_OFFSET_} "\"${_NBL_JSON_DEPENDENCIES_LIST_ELEMENT_CONTENT_}\"")
				endforeach()
			endif()
					
			string(JSON _NBL_GEN_JSON_INPUT_CONFIG_DATA_LENGTH_ ERROR_VARIABLE _NBL_JSON_ERROR_ LENGTH ${_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_} data)
			if(_NBL_GEN_JSON_INPUT_CONFIG_DATA_LENGTH_ AND _NBL_GEN_JSON_INPUT_CONFIG_DATA_LENGTH_ GREATER_EQUAL 1)
				string(JSON _NBL_GEN_JSON_TOP_CONFIG_DATA_LENGTH_ ERROR_VARIABLE _NBL_JSON_ERROR_ LENGTH "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" data)
								
				math(EXPR _NBL_STOP_2_ "${_NBL_GEN_JSON_INPUT_CONFIG_DATA_LENGTH_}-1")

				foreach(_NBL_IDX_2_ RANGE "${_NBL_STOP_2_}")
					math(EXPR _NBL_IDX_2_WITH_OFFSET_ "${_NBL_IDX_2_}+${_NBL_GEN_JSON_TOP_CONFIG_DATA_LENGTH_}")
							
					string(JSON _NBL_JSON_DATA_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_} data ${_NBL_IDX_2_})
					string(JSON "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" data ${_NBL_IDX_2_WITH_OFFSET_} "${_NBL_JSON_DATA_LIST_ELEMENT_CONTENT_}")
				endforeach()
			endif()
		endmacro()

		# an utility to handle inputs field given json array of inputs, the output is concatenated json content
		# from given inputs with all paths referenced validated, resolved to be relative to a generated profile
		# and applied install rules

		# _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ is name of a variable for which json output will be stored to
		# _NBL_JSON_INPUTS_CONTENT_ is json inputs array content
		# _NBL_JSON_INPUTS_FIELD_TRAVERSAL_ is top json traversal to the inputs array field

		function(NBL_JSON_HANDLE_INPUT_CONFIGS _NBL_JSON_GEN_CONTENT_VARIABLE_NAME_ _NBL_JSON_INPUTS_CONTENT_ _NBL_JSON_INPUTS_FIELD_TRAVERSAL_)
			string(JSON _NBL_JSON_INPUTS_TYPE_ ERROR_VARIABLE _NBL_JSON_ERROR_ TYPE "${_NBL_JSON_INPUTS_CONTENT_}")

			if(NOT "${_NBL_JSON_INPUTS_TYPE_}" STREQUAL ARRAY)
				message(FATAL_ERROR "Internal error while processing \"${_NBL_JSON_CONFIG_FILEPATH_}\"! Given _NBL_JSON_INPUTS_CONTENT_ set to \"${_NBL_JSON_INPUTS_CONTENT_}\" is not an ARRAY!")
			endif()

			string(JSON _NBL_JSON_INPUTS_LEN_ ERROR_VARIABLE _NBL_JSON_ERROR_ LENGTH "${_NBL_JSON_INPUTS_CONTENT_}")

			set(${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_} "{}")
			string(JSON "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" dependencies "[]")
			string(JSON "${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}" data "[]")

			if(_NBL_JSON_INPUTS_LEN_ GREATER_EQUAL 1)
				math(EXPR _NBL_STOP_ "${_NBL_JSON_INPUTS_LEN_}-1")
			
				foreach(_NBL_IDX_ RANGE ${_NBL_STOP_})
						NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_INPUTS_FIELD_TRAVERSAL_}[${_NBL_IDX_}]" ${_NBL_IDX_} STRING "${_NBL_JSON_INPUTS_CONTENT_}")
						set(_NBL_JSON_INPUT_CONFIG_FILEPATH_ "${NBL_TEMPLATE_JSON_DIR}/${_NBL_JSON_${_NBL_IDX_}_CONTENT_}")

						if(EXISTS "${_NBL_JSON_INPUT_CONFIG_FILEPATH_}")
							file(READ "${_NBL_JSON_INPUT_CONFIG_FILEPATH_}" _NBL_INPUT_JSON_CONFIG_CONTENT_)
							set(_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_ "${_NBL_INPUT_JSON_CONFIG_CONTENT_}")
					
							NBL_JSON_HANDLE_INPUT_CONFIG_PART(_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_ "${_NBL_JSON_INPUT_CONFIG_FILEPATH_}" "${_NBL_INPUT_JSON_CONFIG_CONTENT_}")
							NBL_JSON_MERGE_INPUT_CONFIGS_CONTENT("${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" "${_NBL_GEN_INPUT_JSON_CONFIG_CONTENT_}")
						else()
							message(FATAL_ERROR "\"${_NBL_JSON_CONFIG_FILEPATH_}\" validation failed! \".${_NBL_JSON_INPUTS_FIELD_TRAVERSAL_}[${_NBL_IDX_}]\" = \"${_NBL_JSON_${_NBL_IDX_}_CONTENT_}\" doesn't exist'. It's filepath is resolved to \"${_NBL_JSON_INPUT_CONFIG_FILEPATH_}\". Note that filepaths in json configs are resolved relative to them.")
						endif()
				endforeach()
			endif()

			set("${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}" 
				"${${_NBL_JSON_GEN_CONTENT_VARIABLE_NAME_}}"
			PARENT_SCOPE)
		endfunction()
		
		# ".inputs" ARRAY
		NBL_JSON_READ_VALIDATE_POPULATE("" inputs ARRAY "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
		NBL_JSON_HANDLE_INPUT_CONFIGS(_NBL_GEN_COMMON_INPUT_JSON_CONFIGS_CONTENT_ "${_NBL_JSON_INPUTS_CONTENT_}" ".inputs")
		
		# ".profiles" array
		NBL_JSON_READ_VALIDATE_POPULATE("" profiles ARRAY "${_NBL_JSON_TOP_CONFIG_CONTENT_}")
			
		if(_NBL_JSON_PROFILES_LEN_ GREATER_EQUAL 1)
			math(EXPR _NBL_STOP_ "${_NBL_JSON_PROFILES_LEN_}-1")
			
			string(JSON _NBL_GEN_JSON_TOP_CONFIG_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ REMOVE "${_NBL_GEN_JSON_TOP_CONFIG_CONTENT_}" profiles)
			
			foreach(_NBL_IDX_ RANGE ${_NBL_STOP_})
				string(JSON _NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ GET ${_NBL_JSON_PROFILES_CONTENT_} ${_NBL_IDX_})
				set(_NBL_JSON_FIELD_TRAVERSAL_ "profiles[${_NBL_IDX_}]")
				
				# "${_NBL_JSON_FIELD_TRAVERSAL_}.backend" string
				NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" backend STRING "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")
				
				# "${_NBL_JSON_FIELD_TRAVERSAL_}.platform" string
				NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" platform STRING "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")
				
				# "${_NBL_JSON_FIELD_TRAVERSAL_}.buildModes" array
				NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" buildModes ARRAY "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")
		
				# "${_NBL_JSON_FIELD_TRAVERSAL_}.runConfiguration" string
				NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" runConfiguration STRING "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")
				
				# "${_NBL_JSON_FIELD_TRAVERSAL_}.gpuArchitectures" array
				NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" gpuArchitectures ARRAY "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")

				# "${_NBL_JSON_FIELD_TRAVERSAL_}.inputs" array
				NBL_JSON_READ_VALIDATE_POPULATE("${_NBL_JSON_FIELD_TRAVERSAL_}" inputs ARRAY "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")
				NBL_JSON_HANDLE_INPUT_CONFIGS(_NBL_GEN_PROFILE_INPUT_JSON_CONFIGS_CONTENT_ "${_NBL_JSON_INPUTS_CONTENT_}" ".${_NBL_JSON_FIELD_TRAVERSAL_}")
				NBL_JSON_MERGE_INPUT_CONFIGS_CONTENT(_NBL_GEN_PROFILE_INPUT_JSON_CONFIGS_CONTENT_ "${_NBL_GEN_COMMON_INPUT_JSON_CONFIGS_CONTENT_}") # merge common json input content to profile's

				string(JSON _NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${_NBL_GEN_JSON_TOP_CONFIG_CONTENT_}" profile "${_NBL_JSON_PROFILES_LIST_ELEMENT_CONTENT_}")

				# move an intput's content to an output profile's global input field object, remove config input reference
				string(JSON _NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ REMOVE "${_NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_}" profile inputs)
				string(JSON _NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ REMOVE "${_NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_}" inputs)
				string(JSON _NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_ ERROR_VARIABLE _NBL_JSON_ERROR_ SET "${_NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_}" input "${_NBL_GEN_PROFILE_INPUT_JSON_CONFIGS_CONTENT_}")

				# after complete validation generate new profile given profiles array, resolve CMake variables and generator expressions according to profile's runConfiguration field (in future more additional conditions may apply)
				string(CONFIGURE "${_NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_}" _NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_CONFIGURED_ ESCAPE_QUOTES)
				
				set(_NBL_JSON_PROFILE_OUTPUT_FILEPATH_ "${NBL_PROFILES_JSON_DIR}/${_NBL_IDX_}.json")
				file(GENERATE OUTPUT "${_NBL_JSON_PROFILE_OUTPUT_FILEPATH_}" CONTENT "${_NBL_GEN_PROFILE_JSON_TOP_CONFIG_CONTENT_CONFIGURED_}" CONDITION $<CONFIG:${_NBL_JSON_RUNCONFIGURATION_CONTENT_}>)
				
				target_sources(${EXECUTABLE_NAME} PUBLIC "${_NBL_JSON_PROFILE_OUTPUT_FILEPATH_}")
				source_group("${NBL_PFT_SOURCE_GROUP_NAME}/target/JSON/Auto-Gen Profiles" FILES "${_NBL_JSON_PROFILE_OUTPUT_FILEPATH_}")
				
				NBL_TEST_MODULE_INSTALL_FILE("${_NBL_JSON_PROFILE_OUTPUT_FILEPATH_}")
			endforeach()
		endif()

		NBL_TARGET_ATTACH_PYTHON_FRAMEWORK("${EXECUTABLE_NAME}")
	endif()
endfunction()

# links builtin resource target to a target
# @_TARGET_@ is target name builtin resource target will be linked to
# @_BS_TARGET_@ is a builtin resource target

function(LINK_BUILTIN_RESOURCES_TO_TARGET _TARGET_ _BS_TARGET_)
	add_dependencies(${_TARGET_} ${_BS_TARGET_})
	target_link_libraries(${_TARGET_} PUBLIC ${_BS_TARGET_})
	
	get_target_property(_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_ ${_BS_TARGET_} BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
	target_include_directories(${_TARGET_} PUBLIC "${_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_}")
endfunction()

macro(nbl_android_create_apk _TARGET)
	get_target_property(TARGET_NAME ${_TARGET} NAME)
	# TARGET_NAME_IDENTIFIER is identifier that can be used in code
	string(MAKE_C_IDENTIFIER ${TARGET_NAME} TARGET_NAME_IDENTIFIER)

	set(APK_FILE_NAME ${TARGET_NAME}.apk)
	set(APK_FILE ${CMAKE_CURRENT_SOURCE_DIR}/bin/$<CONFIG>/${APK_FILE_NAME})
	
	set (variadic_args ${ARGN})
    
    # Did we get any optional args?
    list(LENGTH variadic_args variadic_count)
    if (${variadic_count} GREATER 0)
        list(GET variadic_args 0 optional_arg)
        set(ASSET_SOURCE_DIR ${optional_arg})
		#message(FATAL_ERROR  "the path ${optional_arg} doesn't exist")     
	else()
		set(ASSET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
    endif ()
	
	add_custom_target(${TARGET_NAME}_apk ALL DEPENDS ${APK_FILE})

	string(SUBSTRING
		"${ANDROID_APK_TARGET_ID}"
		8  # length of "android-"
		-1 # take remainder
		TARGET_ANDROID_API_LEVEL
	)
	
	get_filename_component(NBL_GEN_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
	set(NBL_ANDROID_MANIFEST_FILE ${NBL_GEN_DIRECTORY}/$<CONFIG>/AndroidManifest.xml)
	set(NBL_ANDROID_LOADER_JAVA ${NBL_GEN_DIRECTORY}/$<CONFIG>/src/eu/devsh/${TARGET_NAME}/Loader.java)
	
	# AndroidManifest.xml
	add_custom_command(
		OUTPUT "${NBL_ANDROID_MANIFEST_FILE}" 
		COMMAND ${CMAKE_COMMAND} -DNBL_ROOT_PATH:PATH=${NBL_ROOT_PATH} -DNBL_CONFIGURATION:STRING=$<CONFIG> -DNBL_GEN_DIRECTORY:PATH=${NBL_GEN_DIRECTORY} -DTARGET_ANDROID_API_LEVEL:STRING=${TARGET_ANDROID_API_LEVEL} -DSO_NAME:STRING=${_TARGET} -DTARGET_NAME_IDENTIFIER:STRING=${TARGET_NAME_IDENTIFIER} -P ${NBL_ROOT_PATH}/cmake/scripts/nbl/nablaAndroidManifest.cmake #! for some reason CMake fails for OUTPUT_NAME generator expression
		COMMENT "Launching AndroidManifest.xml generation script!"
		VERBATIM
	)
		
	# Loader.java
	add_custom_command(
		OUTPUT "${NBL_ANDROID_LOADER_JAVA}" 
		COMMAND ${CMAKE_COMMAND} -DNBL_ROOT_PATH:PATH=${NBL_ROOT_PATH} -DNBL_CONFIGURATION:STRING=$<CONFIG> -DNBL_GEN_DIRECTORY:PATH=${NBL_GEN_DIRECTORY}/$<CONFIG>/src/eu/devsh/${TARGET_NAME} -DSO_NAME:STRING=${_TARGET} -DTARGET_NAME_IDENTIFIER:STRING=${TARGET_NAME_IDENTIFIER} -P ${NBL_ROOT_PATH}/cmake/scripts/nbl/nablaLoaderJava.cmake
		COMMENT "Launching Loader.java generation script!"
		VERBATIM
	)
	
	# need to sign the apk in order for android device not to refuse it
	set(KEYSTORE_FILE ${NBL_GEN_DIRECTORY}/$<CONFIG>/debug.keystore)
	set(KEY_ENTRY_ALIAS ${TARGET_NAME_IDENTIFIER}_apk_key)
	add_custom_command(
		OUTPUT ${KEYSTORE_FILE}
		WORKING_DIRECTORY ${NBL_GEN_DIRECTORY}/$<CONFIG>
		COMMAND ${ANDROID_JAVA_BIN}/keytool -genkey -keystore ${KEYSTORE_FILE} -storepass android -alias ${KEY_ENTRY_ALIAS} -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=, OU=, O=, L=, S=, C="
	)
	
	if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
		set(D8_SCRIPT "${ANDROID_BUILD_TOOLS}/d8.bat")
		
		if(EXISTS ${D8_SCRIPT})
			set(DEX_COMMAND ${D8_SCRIPT} --output ./bin/ ./obj/eu/devsh/${TARGET_NAME}/*.class)
		else()
			message(FATAL_ERROR "ANDROID_BUILD_TOOLS path doesn't contain D8 (DEX) bat file!")
		endif()
	else()
		set(D8_SCRIPT "${ANDROID_BUILD_TOOLS}/d8")
		
		if(EXISTS ${D8_SCRIPT})
			set(DEX_COMMAND ${D8_SCRIPT} ./obj/eu/devsh/${TARGET_NAME}/Loader.class --output ./bin/)
		else()
			message(FATAL_ERROR "ANDROID_BUILD_TOOLS path doesn't contain D8 (DEX) script file!")
		endif()
	endif()
	
	set(NBL_APK_LIBRARY_DIR libs/lib/x86_64)
	set(NBL_APK_OBJ_DIR obj)
	set(NBL_APK_BIN_DIR bin)
	set(NBL_APK_ASSETS_DIR assets)
	
	if(EXISTS ${ASSET_SOURCE_DIR})
		add_custom_command(
			OUTPUT ${APK_FILE}
			DEPENDS ${_TARGET}
			DEPENDS ${NBL_ANDROID_MANIFEST_FILE}
			DEPENDS ${NBL_ANDROID_LOADER_JAVA}
			DEPENDS ${KEYSTORE_FILE}
			DEPENDS ${NBL_ROOT_PATH}/android/Loader.java
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_LIBRARY_DIR}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_OBJ_DIR}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_BIN_DIR}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_ASSETS_DIR}
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_TARGET}> libs/lib/x86_64/$<TARGET_FILE_NAME:${_TARGET}>
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tbb> libs/lib/x86_64/$<TARGET_FILE_NAME:tbb>
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tbbmalloc> libs/lib/x86_64/$<TARGET_FILE_NAME:tbbmalloc>
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tbbmalloc_proxy> libs/lib/x86_64/$<TARGET_FILE_NAME:tbbmalloc_proxy>
			COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSET_SOURCE_DIR} ${NBL_APK_ASSETS_DIR}
			COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -m -J src -M AndroidManifest.xml -I ${ANDROID_JAR}
			COMMAND ${ANDROID_JAVA_BIN}/javac -d ./obj -source 1.7 -target 1.7 -bootclasspath ${ANDROID_JAVA_RT_JAR} -classpath "${ANDROID_JAR}" -sourcepath src ${NBL_ANDROID_LOADER_JAVA}
			COMMAND ${DEX_COMMAND}
			COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -M AndroidManifest.xml -A ${NBL_APK_ASSETS_DIR} -I ${ANDROID_JAR} -F ${TARGET_NAME}-unaligned.apk bin libs
			COMMAND ${ANDROID_BUILD_TOOLS}/zipalign -f 4 ${TARGET_NAME}-unaligned.apk ${APK_FILE_NAME}
			COMMAND ${ANDROID_BUILD_TOOLS}/apksigner sign --ks ${KEYSTORE_FILE} --ks-pass pass:android --key-pass pass:android --ks-key-alias ${KEY_ENTRY_ALIAS} ${APK_FILE_NAME}
			COMMAND ${CMAKE_COMMAND} -E copy ${APK_FILE_NAME} ${APK_FILE}
			COMMAND ${CMAKE_COMMAND} -E rm -rf ${NBL_APK_ASSETS_DIR}
			WORKING_DIRECTORY ${NBL_GEN_DIRECTORY}/$<CONFIG>
			COMMENT "Creating ${APK_FILE_NAME}..."
			VERBATIM
		)
	else()
		add_custom_command(
			OUTPUT ${APK_FILE}
			DEPENDS ${_TARGET}
			DEPENDS ${NBL_ANDROID_MANIFEST_FILE}
			DEPENDS ${NBL_ANDROID_LOADER_JAVA}
			DEPENDS ${KEYSTORE_FILE}
			DEPENDS ${NBL_ROOT_PATH}/android/Loader.java
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_LIBRARY_DIR}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_OBJ_DIR}
			COMMAND ${CMAKE_COMMAND} -E make_directory ${NBL_APK_BIN_DIR}
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_TARGET}> libs/lib/x86_64/$<TARGET_FILE_NAME:${_TARGET}>
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tbb> libs/lib/x86_64/$<TARGET_FILE_NAME:tbb>
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tbbmalloc> libs/lib/x86_64/$<TARGET_FILE_NAME:tbbmalloc>
			COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tbbmalloc_proxy> libs/lib/x86_64/$<TARGET_FILE_NAME:tbbmalloc_proxy>
			COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -m -J src -M AndroidManifest.xml -I ${ANDROID_JAR}
			COMMAND ${ANDROID_JAVA_BIN}/javac -d ./obj -source 1.7 -target 1.7 -bootclasspath ${ANDROID_JAVA_RT_JAR} -classpath "${ANDROID_JAR}" -sourcepath src ${NBL_ANDROID_LOADER_JAVA}
			COMMAND ${DEX_COMMAND}
			COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -M AndroidManifest.xml -I ${ANDROID_JAR} -F ${TARGET_NAME}-unaligned.apk bin libs
			COMMAND ${ANDROID_BUILD_TOOLS}/zipalign -f 4 ${TARGET_NAME}-unaligned.apk ${APK_FILE_NAME}
			COMMAND ${ANDROID_BUILD_TOOLS}/apksigner sign --ks ${KEYSTORE_FILE} --ks-pass pass:android --key-pass pass:android --ks-key-alias ${KEY_ENTRY_ALIAS} ${APK_FILE_NAME}
			COMMAND ${CMAKE_COMMAND} -E copy ${APK_FILE_NAME} ${APK_FILE}
			WORKING_DIRECTORY ${NBL_GEN_DIRECTORY}/$<CONFIG>
			COMMENT "Creating ${APK_FILE_NAME}..."
			VERBATIM
		)
	endif()
endmacro()

function(nbl_android_create_media_storage_apk)
	set(TARGET_NAME android_media_storage)
	string(MAKE_C_IDENTIFIER ${TARGET_NAME} TARGET_NAME_IDENTIFIER)

	set(APK_FILE_NAME ${TARGET_NAME}.apk)
	set(APK_FILE ${CMAKE_CURRENT_BINARY_DIR}/media_storage/bin/${APK_FILE_NAME})

	add_custom_target(${TARGET_NAME}_apk ALL DEPENDS ${APK_FILE})

	string(SUBSTRING
		"${ANDROID_APK_TARGET_ID}"
		8  # length of "android-"
		-1 # take remainder
		TARGET_ANDROID_API_LEVEL
	)
	set(PACKAGE_NAME "eu.devsh.${TARGET_NAME_IDENTIFIER}")
	set(APP_NAME ${TARGET_NAME_IDENTIFIER})

	# configure_file(${NBL_ROOT_PATH}/android/AndroidManifest.xml ${CMAKE_CURRENT_BINARY_DIR}/AndroidManifest.xml)

	# # need to sign the apk in order for android device not to refuse it
	# set(KEYSTORE_FILE ${CMAKE_CURRENT_BINARY_DIR}/debug.keystore)
	# set(KEY_ENTRY_ALIAS ${TARGET_NAME_IDENTIFIER}_apk_key)
	# add_custom_command(
		# OUTPUT ${KEYSTORE_FILE}
		# WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		# COMMAND ${ANDROID_JAVA_BIN}/keytool -genkey -keystore ${KEYSTORE_FILE} -storepass android -alias ${KEY_ENTRY_ALIAS} -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=, OU=, O=, L=, S=, C="
	# )
	
	 add_custom_command(
		OUTPUT ${APK_FILE}
		DEPENDS ${KEYSTORE_FILE}
		DEPENDS ${NBL_ROOT_PATH}/android/AndroidManifest.xml
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/AndroidManifest.xml
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Creating ${APK_FILE_NAME} ..."
		COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -m -J src -M AndroidManifest.xml -I ${ANDROID_JAR}
		COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -M AndroidManifest.xml -I ${ANDROID_JAR} -F ${TARGET_NAME}-unaligned.apk ${CMAKE_CURRENT_SOURCE_DIR}/media
		COMMAND ${ANDROID_BUILD_TOOLS}/zipalign -f 4 ${TARGET_NAME}-unaligned.apk ${APK_FILE_NAME}
		COMMAND ${ANDROID_BUILD_TOOLS}/apksigner sign --ks ${KEYSTORE_FILE} --ks-pass pass:android --key-pass pass:android --ks-key-alias ${KEY_ENTRY_ALIAS} ${APK_FILE_NAME}
		COMMAND ${CMAKE_COMMAND} -E copy ${APK_FILE_NAME} ${APK_FILE}
		VERBATIM
	 )
endfunction()

# Start to track variables for change or adding.
# Note that variables starting with underscore are ignored.
macro(start_tracking_variables_for_propagation_to_parent)
    get_cmake_property(_fnvtps_cache_vars CACHE_VARIABLES)
    get_cmake_property(_fnvtps_old_vars VARIABLES)
    
    foreach(_i ${_fnvtps_old_vars})
        if (NOT "x${_i}" MATCHES "^x_.*$")
            list(FIND _fnvtps_cache_vars ${_i} _fnvtps_is_in_cache)
            if(${_fnvtps_is_in_cache} EQUAL -1)
                set("_fnvtps_old${_i}" ${${_i}})
                #message(STATUS "_fnvtps_old${_i} = ${_fnvtps_old${_i}}")
            endif()
        endif()
    endforeach()
endmacro()

# forward_changed_variables_to_parent_scope([exclusions])
# Forwards variables that was added/changed since last call to start_track_variables() to the parent scope.
# Note that variables starting with underscore are ignored.
macro(propagate_changed_variables_to_parent_scope)
    get_cmake_property(_fnvtps_cache_vars CACHE_VARIABLES)
    get_cmake_property(_fnvtps_vars VARIABLES)
    set(_fnvtps_cache_vars ${_fnvtps_cache_vars} ${ARGN})
    
    foreach(_i ${_fnvtps_vars})
        if (NOT "x${_i}" MATCHES "^x_.*$")
            list(FIND _fnvtps_cache_vars ${_i} _fnvtps_is_in_cache)
            
            if (${_fnvtps_is_in_cache} EQUAL -1)
                list(FIND _fnvtps_old_vars ${_i} _fnvtps_is_old)
                
                if(${_fnvtps_is_old} EQUAL -1 OR NOT "${${_i}}" STREQUAL "${_fnvtps_old${_i}}")
                    set(${_i} ${${_i}} PARENT_SCOPE)
                    #message(STATUS "forwarded var ${_i}")
                endif()
            endif()
        endif()
    endforeach()
endmacro()

function(NBL_GET_ALL_TARGETS NBL_OUTPUT_VAR)
    set(NBL_TARGETS)
    NBL_GET_ALL_TARGETS_RECURSIVE(NBL_TARGETS ${CMAKE_CURRENT_SOURCE_DIR})
    set(${NBL_OUTPUT_VAR} ${NBL_TARGETS} PARENT_SCOPE)
endfunction()

macro(NBL_GET_ALL_TARGETS_RECURSIVE NBL_TARGETS NBL_DIRECTORY)
    get_property(NBL_SUBDIRECTORIES DIRECTORY ${NBL_DIRECTORY} PROPERTY SUBDIRECTORIES)
    foreach(NBL_SUBDIRECTORY ${NBL_SUBDIRECTORIES})
        NBL_GET_ALL_TARGETS_RECURSIVE(${NBL_TARGETS} ${NBL_SUBDIRECTORY})
    endforeach()

    get_property(NBL_GATHERED_TARGETS DIRECTORY ${NBL_DIRECTORY} PROPERTY BUILDSYSTEM_TARGETS)
    list(APPEND ${NBL_TARGETS} ${NBL_GATHERED_TARGETS})
endmacro()

function(NBL_IMPORT_VS_CONFIG)
	if(WIN32 AND "${CMAKE_GENERATOR}" MATCHES "Visual Studio")
		message(STATUS "Requesting import of .vsconfig file! Configuration will continue after Visual Studio Installer is closed.")
		set(NBL_DEVENV_ISOLATION_INI_PATH "${CMAKE_GENERATOR_INSTANCE}/Common7/IDE/devenv.isolation.ini")
		file(READ ${NBL_DEVENV_ISOLATION_INI_PATH} NBL_DEVENV_ISOLATION_INI_CONTENT)
		string(REPLACE "/" "\\" NBL_VS_INSTALLATION_PATH ${CMAKE_GENERATOR_INSTANCE})
		string(REGEX MATCH "SetupEngineFilePath=\"([^\"]*)\"" _match "${NBL_DEVENV_ISOLATION_INI_CONTENT}")
		set(NBL_VS_INSTALLER_PATH "${CMAKE_MATCH_1}")

		execute_process(COMMAND "${NBL_VS_INSTALLER_PATH}" modify --installPath "${NBL_VS_INSTALLATION_PATH}" --config "${NBL_ROOT_PATH}/.vsconfig" --allowUnsignedExtensions
			ERROR_VARIABLE vsconfig_error
			RESULT_VARIABLE vsconfig_result
		)

		if(NOT vsconfig_result EQUAL 0)
    		message(FATAL_ERROR "Visual Studio Installer error: ${vsconfig_error}")
		endif()
	else()
		message(FATAL_ERORR "Cannot request importing VS config, doesn't meet requirements!")
	endif()

endfunction()

macro(NBL_TARGET_FORCE_ASSEMBLER_EXECUTABLE _NBL_TARGET_ _NBL_ASM_DIALECT_ _NBL_PREPEND_PATH_TRANSFORM_)
	get_target_property(_NBL_TARGET_SOURCES_ "${_NBL_TARGET_}" SOURCES)
	list(FILTER _NBL_TARGET_SOURCES_ INCLUDE REGEX "\\.asm$")
	list(TRANSFORM _NBL_TARGET_SOURCES_ PREPEND "${_NBL_PREPEND_PATH_TRANSFORM_}")

	set_source_files_properties(${_NBL_TARGET_SOURCES_}
		TARGET_DIRECTORY "${_NBL_TARGET_}"
		PROPERTIES LANGUAGE "${_NBL_ASM_DIALECT_}"
	)
endmacro()

macro(NBL_WAIT_FOR SLEEP_DURATION)
	execute_process(COMMAND ${CMAKE_COMMAND} -E sleep ${SLEEP_DURATION})
endmacro()

macro(NBL_DOCKER)
	execute_process(COMMAND ${DOCKER_EXECUTABLE} ${ARGN} 
		RESULT_VARIABLE DOCKER_EXIT_CODE 
		OUTPUT_VARIABLE DOCKER_OUTPUT_VAR
	)
endmacro()

function(NBL_ADJUST_FOLDERS NS)
	NBL_GET_ALL_TARGETS(TARGETS)
	foreach(T IN LISTS TARGETS)
		get_target_property(NBL_FOLDER ${T} FOLDER)
				
		if(NBL_FOLDER)
			set_target_properties(${T} PROPERTIES FOLDER "${NS}/${NBL_FOLDER}")
		else()
			set_target_properties(${T} PROPERTIES FOLDER "${NS}")
		endif()
	endforeach()
endfunction()

function(NBL_PARSE_REQUIRED PREFIX)
	set(VARIADIC ${ARGV})
	list(POP_FRONT VARIADIC VARIADIC)
	foreach(ARG ${VARIADIC})
		set(V ${PREFIX}_${ARG})
		if(NOT ${V})
			message(FATAL_ERROR "\"${ARG}\" argument missing!")
		endif()
	endforeach()
endfunction()

# TODO: could create them in the function as <TARGET>_<PROPERTY> properties

define_property(SOURCE PROPERTY NBL_SPIRV_REGISTERED_INPUT
	BRIEF_DOCS "Absolute path to input shader which will be glued with device permutation config caps auto-gen file using #include directive, used as part of NSC compile rule to produce SPIRV output"
)

define_property(SOURCE PROPERTY NBL_SPIRV_PERMUTATION_CONFIG
	BRIEF_DOCS "Absolute path to intermediate config file, used as part of NSC compile rule to produce SPIRV output"
	FULL_DOCS "The file is auto-generated at configuration time, contains DeviceConfigCaps struct with permuted device caps after which #include directive glues it with an input shader"
)

define_property(SOURCE PROPERTY NBL_SPIRV_BINARY_DIR
	BRIEF_DOCS "a <SPIRV output> = NBL_SPIRV_BINARY_DIR/NBL_SPIRV_ACCESS_KEY"
)
define_property(SOURCE PROPERTY NBL_SPIRV_ACCESS_KEY
	BRIEF_DOCS "a <SPIRV output> = NBL_SPIRV_BINARY_DIR/NBL_SPIRV_ACCESS_KEY"
)

define_property(TARGET PROPERTY NBL_CANONICAL_IDENTIFIERS
	BRIEF_DOCS "List of identifiers composed as NBL_SPIRV_BINARY_DIR/KEY"
	FULL_DOCS "For a given NBL_SPIRV_BINARY_DIR we define a set of canonical KEYs, each unique given which at runtime one can get SPIRV key to access <SPIRV output> by the canonical key which may contain special permutation part in character of <key>(.<name>=<value>)(.<name>=<value>)(...)"
)

define_property(TARGET PROPERTY NBL_SPIRV_OUTPUTS
	BRIEF_DOCS "List of absolute paths to all <SPIRV output>s which are part of NSC compile rules"
)

define_property(TARGET PROPERTY NBL_HEADER_PATH
	BRIEF_DOCS "Relative path for auto-gen include file with key getters"
)
define_property(TARGET PROPERTY NBL_HEADER_GENERATED_RULE)

define_property(TARGET PROPERTY NBL_HEADER_CONTENT
	BRIEF_DOCS "Contains NBL_HEADER_PATH's content"
)

define_property(TARGET PROPERTY NBL_MOUNT_POINT_DEFINES
	BRIEF_DOCS "List of preprocessor defines with mount points"
)

option(NSC_DEBUG_EDIF_FILE_BIT "Add \"-fspv-debug=file\" to NSC Debug CLI" ON)
option(NSC_DEBUG_EDIF_SOURCE_BIT "Add \"-fspv-debug=source\" to NSC Debug CLI" OFF)
option(NSC_DEBUG_EDIF_LINE_BIT "Add \"-fspv-debug=line\" to NSC Debug CLI" OFF)
option(NSC_DEBUG_EDIF_TOOL_BIT "Add \"-fspv-debug=tool\" to NSC Debug CLI" ON)
option(NSC_DEBUG_EDIF_NON_SEMANTIC_BIT "Add \"-fspv-debug=vulkan-with-source\" to NSC Debug CLI" OFF)

function(NBL_CREATE_NSC_COMPILE_RULES)
    set(COMMENT "this code has been autogenerated with Nabla CMake NBL_CREATE_HLSL_COMPILE_RULES utility")
    set(DEVICE_CONFIG_VIEW
[=[

// -> @COMMENT@!
#ifndef _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
#define _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
struct DeviceConfigCaps
{
@CAPS_EVAL@
};

#include "@TARGET_INPUT@"

#endif // __HLSL_VERSION
#endif // _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
// <- @COMMENT@!

]=])

	# would get added by NSC anyway and spam in output
	set(REQUIRED_OPTIONS
		-HV 202x 
		-Wno-c++14-extensions 
		-Wno-gnu-static-float-init 
		-Wno-c++1z-extensions 
		-Wno-c++11-extensions 
		-fvk-use-scalar-layout 
		-enable-16bit-types 
		-Zpr 
		-spirv 
		-Wno-local-type-template-args
		-fspv-target-env=vulkan1.3 
		-WShadow 
		-WConversion 
		$<$<CONFIG:Debug>:-O0> 
		$<$<CONFIG:Release>:-O3> 
		$<$<CONFIG:RelWithDebInfo>:-O3>
	)

	if(NSC_DEBUG_EDIF_FILE_BIT)
    	list(APPEND REQUIRED_OPTIONS $<$<CONFIG:Debug>:-fspv-debug=file>)
	endif()
	
	if(NSC_DEBUG_EDIF_SOURCE_BIT)
	    list(APPEND REQUIRED_OPTIONS $<$<CONFIG:Debug>:-fspv-debug=source>)
	endif()
	
	if(NSC_DEBUG_EDIF_LINE_BIT)
	    list(APPEND REQUIRED_OPTIONS $<$<CONFIG:Debug>:-fspv-debug=line>)
	endif()
	
	if(NSC_DEBUG_EDIF_TOOL_BIT)
	    list(APPEND REQUIRED_OPTIONS $<$<CONFIG:Debug>:-fspv-debug=tool>)
	endif()
	
	if(NSC_DEBUG_EDIF_NON_SEMANTIC_BIT)
	    list(APPEND REQUIRED_OPTIONS $<$<CONFIG:Debug>:-fspv-debug=vulkan-with-source>)
	endif()

	if(NOT NBL_EMBED_BUILTIN_RESOURCES)
		list(APPEND REQUIRED_OPTIONS
			-I "${NBL_ROOT_PATH}/include"
			-I "${NBL_ROOT_PATH}/3rdparty/dxc/dxc/external/SPIRV-Headers/include"
			-I "${NBL_ROOT_PATH}/3rdparty/boost/superproject/libs/preprocessor/include"
			-I "${NBL_ROOT_PATH_BINARY}/src/nbl/device/include"
		)
	endif()

    set(REQUIRED_SINGLE_ARGS TARGET BINARY_DIR OUTPUT_VAR INPUTS INCLUDE NAMESPACE MOUNT_POINT_DEFINE)
    cmake_parse_arguments(IMPL "" "${REQUIRED_SINGLE_ARGS};LINK_TO" "COMMON_OPTIONS;DEPENDS" ${ARGV})
    NBL_PARSE_REQUIRED(IMPL ${REQUIRED_SINGLE_ARGS})

	if(NOT TARGET ${IMPL_TARGET})
		add_library(${IMPL_TARGET} INTERFACE)
	endif()

	if(IMPL_LINK_TO)
		target_link_libraries(${IMPL_LINK_TO} PUBLIC ${IMPL_TARGET})
	endif()

	if(IS_ABSOLUTE "${IMPL_INCLUDE}")
		message(FATAL_ERROR "INCLUDE argument must be relative path")
	endif()

	set_target_properties(${IMPL_TARGET} PROPERTIES NBL_HEADER_PATH "${IMPL_INCLUDE}")

	get_target_property(HEADER_RULE_GENERATED ${IMPL_TARGET} NBL_HEADER_GENERATED_RULE)
	if(NOT HEADER_RULE_GENERATED)
	    set(INCLUDE_DIR "$<TARGET_PROPERTY:${IMPL_TARGET},BINARY_DIR>/${IMPL_TARGET}/.cmake/include/$<CONFIG>")
		set(INCLUDE_FILE "${INCLUDE_DIR}/$<TARGET_PROPERTY:${IMPL_TARGET},NBL_HEADER_PATH>")
		set(INCLUDE_CONTENT $<TARGET_PROPERTY:${IMPL_TARGET},NBL_HEADER_CONTENT>)

		file(GENERATE OUTPUT ${INCLUDE_FILE}
			CONTENT $<GENEX_EVAL:${INCLUDE_CONTENT}>
			TARGET ${IMPL_TARGET}
		)

		target_sources(${IMPL_TARGET} PUBLIC ${INCLUDE_FILE})
		set_source_files_properties(${INCLUDE_FILE} PROPERTIES 
			HEADER_FILE_ONLY ON
			VS_TOOL_OVERRIDE None
		)

		target_compile_definitions(${IMPL_TARGET} INTERFACE $<TARGET_PROPERTY:${IMPL_TARGET},NBL_MOUNT_POINT_DEFINES>)
		target_include_directories(${IMPL_TARGET} INTERFACE ${INCLUDE_DIR})
		set_target_properties(${IMPL_TARGET} PROPERTIES NBL_HEADER_GENERATED_RULE ON)

		set(HEADER_ITEM_VIEW [=[
#include "nabla.h"

]=])
		set_property(TARGET ${IMPL_TARGET} APPEND_STRING PROPERTY NBL_HEADER_CONTENT "${HEADER_ITEM_VIEW}")
	endif()

	string(MAKE_C_IDENTIFIER "${IMPL_TARGET}_${IMPL_NAMESPACE}" NS_IMPL_KEYS_PROPERTY)
	get_property(NS_IMPL_KEYS_PROPERTY_DEFINED
		TARGET    ${IMPL_TARGET}
		PROPERTY "${NS_IMPL_KEYS_PROPERTY}"
		DEFINED
	)
	if(NOT NS_IMPL_KEYS_PROPERTY_DEFINED)
		set(HEADER_ITEM_VIEW [=[
namespace @IMPL_NAMESPACE@ {
	template<nbl::core::StringLiteral Key>
	inline const nbl::core::string get_spirv_key(const nbl::video::SPhysicalDeviceLimits& limits, const nbl::video::SPhysicalDeviceFeatures& features);

	template<nbl::core::StringLiteral Key>
	inline const nbl::core::string get_spirv_key(const nbl::video::ILogicalDevice* device)
	{
		return get_spirv_key<Key>(device->getPhysicalDevice()->getLimits(), device->getEnabledFeatures());
	}
}

]=])
		string(CONFIGURE "${HEADER_ITEM_VIEW}" HEADER_ITEM_EVAL @ONLY)
		set_property(TARGET ${IMPL_TARGET} APPEND_STRING PROPERTY NBL_HEADER_CONTENT "${HEADER_ITEM_EVAL}")
		define_property(TARGET PROPERTY "${NS_IMPL_KEYS_PROPERTY}")
	endif()

	get_target_property(MP_DEFINES ${IMPL_TARGET} NBL_MOUNT_POINT_DEFINES)
	if(NOT MP_DEFINES)
		unset(MP_DEFINES)
	endif()
	list(FILTER MP_DEFINES EXCLUDE REGEX "^${IMPL_MOUNT_POINT_DEFINE}=")
	list(APPEND MP_DEFINES ${IMPL_MOUNT_POINT_DEFINE}="${IMPL_BINARY_DIR}")
	set_target_properties(${IMPL_TARGET} PROPERTIES NBL_MOUNT_POINT_DEFINES "${MP_DEFINES}")

    string(JSON JSON_LENGTH LENGTH "${IMPL_INPUTS}")
    math(EXPR LAST_INDEX "${JSON_LENGTH} - 1")

    set(ALL_OUTPUT_KEYS "")

    foreach(INDEX RANGE ${LAST_INDEX})
        string(JSON INPUT GET "${IMPL_INPUTS}" ${INDEX} INPUT)
		string(JSON BASE_KEY GET "${IMPL_INPUTS}" ${INDEX} KEY)
        
        set(COMPILE_OPTIONS "")
		string(JSON HAS_COMPILE_OPTIONS ERROR_VARIABLE ERROR_VAR TYPE "${IMPL_INPUTS}" ${INDEX} COMPILE_OPTIONS)
		if(HAS_COMPILE_OPTIONS STREQUAL "ARRAY")
			string(JSON COMPILE_OPTIONS_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} COMPILE_OPTIONS)
			if(NOT COMPILE_OPTIONS_LENGTH EQUAL 0)
				math(EXPR LAST_CO "${COMPILE_OPTIONS_LENGTH} - 1")
				foreach(COMP_IDX RANGE 0 ${LAST_CO})
					string(JSON COMP_ITEM GET "${IMPL_INPUTS}" ${INDEX} COMPILE_OPTIONS ${COMP_IDX})
					list(APPEND COMPILE_OPTIONS "${COMP_ITEM}")
				endforeach()
			endif()
		endif()

		set(DEPENDS_ON "")
        string(JSON HAS_DEPENDS ERROR_VARIABLE ERROR_VAR TYPE "${IMPL_INPUTS}" ${INDEX} DEPENDS)
        if(HAS_DEPENDS STREQUAL "ARRAY")
            string(JSON DEPENDS_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} DEPENDS)
            if(NOT DEPENDS_LENGTH EQUAL 0)
                math(EXPR LAST_DEP "${DEPENDS_LENGTH} - 1")
                foreach(DEP_IDX RANGE 0 ${LAST_DEP})
                    string(JSON DEP_ITEM GET "${IMPL_INPUTS}" ${INDEX} DEPENDS ${DEP_IDX})
                    list(APPEND DEPENDS_ON "${DEP_ITEM}")
                endforeach()
            endif()
        endif()

		if(IMPL_DEPENDS)
			list(APPEND DEPENDS_ON ${IMPL_DEPENDS})
		endif()

        set(HAS_CAPS FALSE)
        set(CAPS_LENGTH 0)
        string(JSON CAPS_TYPE ERROR_VARIABLE ERROR_VAR TYPE "${IMPL_INPUTS}" ${INDEX} CAPS)
        if(CAPS_TYPE STREQUAL "ARRAY")
            string(JSON CAPS_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} CAPS)
            if(NOT CAPS_LENGTH EQUAL 0)
                set(HAS_CAPS TRUE)
            endif()
        endif()

		function(ERROR_WHILE_PARSING_ITEM)
			string(JSON ITEM GET "${IMPL_INPUTS}" ${INDEX})
			message(FATAL_ERROR 
				"While parsing ${IMPL_TARGET}'s NSC compile rule\n${ITEM}\n"
				${ARGV}
			)
		endfunction()

        set(CAP_NAMES "")
        set(CAP_TYPES "")
		set(CAP_KINDS "")
        if(HAS_CAPS)
            math(EXPR LAST_CAP "${CAPS_LENGTH} - 1")
            foreach(CAP_IDX RANGE 0 ${LAST_CAP})
				string(JSON CAP_KIND ERROR_VARIABLE CAP_TYPE_ERROR GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} kind)
                string(JSON CAP_NAME GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} name)
                string(JSON CAP_TYPE GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} type)

				# -> TODO: improve validation, input should be string
				if(CAP_TYPE_ERROR)
					set(CAP_KIND limits) # I assume its limit by default (or when invalid value present, currently)
				else()
					if(NOT CAP_KIND MATCHES "^(limits|features)$")
						ERROR_WHILE_PARSING_ITEM(
							"Invalid CAP kind \"${CAP_KIND}\" for ${CAP_NAME}\n"
							"Allowed kinds are: limits, features"
						)
					endif()
				endif()
				# <-

				if(NOT CAP_TYPE MATCHES "^(bool|uint16_t|uint32_t|uint64_t)$")
					ERROR_WHILE_PARSING_ITEM(
						"Invalid CAP type \"${CAP_TYPE}\" for ${CAP_NAME}\n"
						"Allowed types are: bool, uint16_t, uint32_t, uint64_t"
					)
				endif()

				string(JSON CAP_VALUES_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} values)

				set(VALUES "")
				math(EXPR LAST_VAL "${CAP_VALUES_LENGTH} - 1")
				foreach(VAL_IDX RANGE 0 ${LAST_VAL})
					string(JSON VALUE GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} values ${VAL_IDX})
					string(JSON VAL_TYPE TYPE "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} values ${VAL_IDX})

					if(NOT VAL_TYPE STREQUAL "NUMBER")
						ERROR_WHILE_PARSING_ITEM(
							"Invalid CAP value \"${VALUE}\" for CAP \"${CAP_NAME}\" of type ${CAP_TYPE}\n"
							"Use numbers for uint*_t and 0/1 for bools."
						)
					endif()

					if(CAP_TYPE STREQUAL "bool")
						if(NOT VALUE MATCHES "^[01]$")
							ERROR_WHILE_PARSING_ITEM(
								"Invalid bool value \"${VALUE}\" for ${CAP_NAME}\n"
								"Boolean CAPs can only have values 0 or 1."
							)
						endif()
					endif()

					list(APPEND VALUES "${VALUE}")
				endforeach()

                set(CAP_VALUES_${CAP_IDX} "${VALUES}")
                list(APPEND CAP_NAMES "${CAP_NAME}")
                list(APPEND CAP_TYPES "${CAP_TYPE}")
				list(APPEND CAP_KINDS "${CAP_KIND}")
            endforeach()
        endif()

        list(LENGTH CAP_NAMES NUM_CAPS)

		set(TARGET_INPUT "${INPUT}")
		if(NOT IS_ABSOLUTE "${TARGET_INPUT}")
			set(TARGET_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_INPUT}")
		endif()

		get_target_property(CANONICAL_IDENTIFIERS ${IMPL_TARGET} NBL_CANONICAL_IDENTIFIERS)

		set(NEW_CANONICAL_IDENTIFIER "${IMPL_BINARY_DIR}/${BASE_KEY}")
		if(CANONICAL_IDENTIFIERS)
			list(FIND CANONICAL_IDENTIFIERS "${NEW_CANONICAL_IDENTIFIER}" FOUND)

			if(NOT FOUND STREQUAL -1)
				string(JSON ITEM GET "${IMPL_INPUTS}" ${INDEX})
				message(FATAL_ERROR "While parsing ${IMPL_TARGET}'s NSC compile rule\n${ITEM}\nwith binary directory \"${IMPL_BINARY_DIR}\",\ncanonical key \"${BASE_KEY}\" already defined!")
			endif()
		endif()

		set_property(TARGET ${IMPL_TARGET} APPEND PROPERTY NBL_CANONICAL_IDENTIFIERS "${NEW_CANONICAL_IDENTIFIER}")

		set(HEADER_ITEM_VIEW [=[
namespace @IMPL_NAMESPACE@ {
	template<>
	inline const nbl::core::string get_spirv_key<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>
	(const nbl::video::SPhysicalDeviceLimits& limits, const nbl::video::SPhysicalDeviceFeatures& features)
	{
		nbl::core::string retval = "@BASE_KEY@";
@RETVAL_EVAL@
		retval += ".spv";
		return "$<CONFIG>/" + retval;
	}
}

]=])
		unset(RETVAL_EVAL)
		list(LENGTH CAP_NAMES CAP_COUNT)
		if(CAP_COUNT GREATER 0)
			math(EXPR LAST_CAP "${CAP_COUNT} - 1")
			foreach(i RANGE ${LAST_CAP})
				list(GET CAP_NAMES ${i} CAP)
				list(GET CAP_KINDS ${i} KIND)
				string(CONFIGURE [=[
		retval += ".@CAP@_" + std::to_string(@KIND@.@CAP@);
]=] 			RETVALUE_VIEW @ONLY)
				string(APPEND RETVAL_EVAL "${RETVALUE_VIEW}")
			endforeach()
		endif()
		
		string(CONFIGURE "${HEADER_ITEM_VIEW}" HEADER_ITEM_EVAL @ONLY)
		set_property(TARGET ${IMPL_TARGET} APPEND_STRING PROPERTY NBL_HEADER_CONTENT "${HEADER_ITEM_EVAL}")
		
		function(GENERATE_KEYS PREFIX CAP_INDEX CAPS_EVAL_PART)
			if(NUM_CAPS EQUAL 0 OR CAP_INDEX EQUAL ${NUM_CAPS})
			# generate .config file
				set(FINAL_KEY "${BASE_KEY}${PREFIX}.spv") # always add ext even if its already there to make sure asset loader always is able to load as IShader
				set(CONFIG_FILE_TARGET_OUTPUT "${IMPL_BINARY_DIR}/${FINAL_KEY}")
				set(CONFIG_FILE "${CONFIG_FILE_TARGET_OUTPUT}.config")
				set(CAPS_EVAL "${CAPS_EVAL_PART}")
				string(CONFIGURE "${DEVICE_CONFIG_VIEW}" CONFIG_CONTENT @ONLY)
				file(WRITE "${CONFIG_FILE}" "${CONFIG_CONTENT}")

				# generate keys and commands for compiling shaders
				foreach(BUILD_CONFIGURATION ${CMAKE_CONFIGURATION_TYPES})
					set(FINAL_KEY_REL_PATH "${BUILD_CONFIGURATION}/${FINAL_KEY}")
					set(TARGET_OUTPUT "${IMPL_BINARY_DIR}/${FINAL_KEY_REL_PATH}")

					set(NBL_NSC_COMPILE_COMMAND
						"$<TARGET_FILE:nsc>"
						-Fc "${TARGET_OUTPUT}"
						${COMPILE_OPTIONS} ${REQUIRED_OPTIONS} ${IMPL_COMMON_OPTIONS}
						"${CONFIG_FILE}"
					)

					add_custom_command(OUTPUT "${TARGET_OUTPUT}"
						COMMAND ${NBL_NSC_COMPILE_COMMAND}
						DEPENDS ${DEPENDS_ON}
						COMMENT "Creating \"${TARGET_OUTPUT}\""
						VERBATIM
						COMMAND_EXPAND_LISTS
					)

					set(HEADER_ONLY_LIKE "${CONFIG_FILE}" "${TARGET_INPUT}" "${TARGET_OUTPUT}")
					target_sources(${IMPL_TARGET} PRIVATE ${HEADER_ONLY_LIKE})

					set_source_files_properties(${HEADER_ONLY_LIKE} PROPERTIES 
						HEADER_FILE_ONLY ON
						VS_TOOL_OVERRIDE None
					)

					set_source_files_properties("${TARGET_OUTPUT}" PROPERTIES
						NBL_SPIRV_REGISTERED_INPUT "${TARGET_INPUT}"
						NBL_SPIRV_PERMUTATION_CONFIG "${CONFIG_FILE}"
						NBL_SPIRV_BINARY_DIR "${IMPL_BINARY_DIR}"
						NBL_SPIRV_ACCESS_KEY "${FINAL_KEY_REL_PATH}"
					)

					set_property(TARGET ${IMPL_TARGET} APPEND PROPERTY NBL_SPIRV_OUTPUTS "${TARGET_OUTPUT}")
					endforeach()
				return()
			endif()

			list(GET CAP_NAMES ${CAP_INDEX} CURRENT_CAP)
			list(GET CAP_TYPES ${CAP_INDEX} CURRENT_TYPE)
			list(GET CAP_KINDS ${CAP_INDEX} CURRENT_KIND)
			set(VAR_NAME "CAP_VALUES_${CAP_INDEX}")
			set(VALUES "${${VAR_NAME}}")

			foreach(V IN LISTS VALUES)
				set(NEW_PREFIX "${PREFIX}.${CURRENT_CAP}_${V}")
				set(NEW_EVAL "${CAPS_EVAL_PART}NBL_CONSTEXPR_STATIC_INLINE ${CURRENT_TYPE} ${CURRENT_CAP} = (${CURRENT_TYPE}) ${V}; // got permuted\n")
				math(EXPR NEXT_INDEX "${CAP_INDEX} + 1")
				GENERATE_KEYS("${NEW_PREFIX}" "${NEXT_INDEX}" "${NEW_EVAL}")
			endforeach()
		endfunction()

       	GENERATE_KEYS("" 0 "")
    endforeach()

	unset(KEYS)
	get_target_property(SPIRVs ${IMPL_TARGET} NBL_SPIRV_OUTPUTS)
	foreach(SPIRV ${SPIRVs})
		get_source_file_property(CONFIG ${SPIRV} NBL_SPIRV_PERMUTATION_CONFIG)
		get_source_file_property(INPUT ${SPIRV} NBL_SPIRV_REGISTERED_INPUT)
		get_source_file_property(ACCESS_KEY ${SPIRV} NBL_SPIRV_ACCESS_KEY)

		list(APPEND CONFIGS ${CONFIG})
		list(APPEND INPUTS ${INPUT})
		list(APPEND KEYS ${ACCESS_KEY})
	endforeach()

	set(RTE "NSC Rules")
	set(IN "${RTE}/In")
	set(OUT "${RTE}/Out")

	source_group("${IN}" FILES ${CONFIGS} ${INPUTS})
	source_group("${OUT}" FILES ${SPIRVs})

	set(${IMPL_OUTPUT_VAR} ${KEYS} PARENT_SCOPE)
endfunction()

function(NBL_CREATE_RESOURCE_ARCHIVE)
    set(REQUIRED_SINGLE_ARGS TARGET BIND NAMESPACE)
    cmake_parse_arguments(IMPL "" "${REQUIRED_SINGLE_ARGS}" "BUILTINS;LINK_TO" ${ARGV})
    NBL_PARSE_REQUIRED(IMPL ${REQUIRED_SINGLE_ARGS})

	if(NOT NBL_EMBED_BUILTIN_RESOURCES)
		add_library(${IMPL_TARGET} INTERFACE) # dummy, could use LINK_TO but makes no difference in this case
		return()
	endif()

	set(IMPL_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${IMPL_TARGET}")

	set(_BUNDLE_ARCHIVE_ABSOLUTE_PATH_ "")
	get_filename_component(_BUNDLE_SEARCH_DIRECTORY_ "${IMPL_BIND}" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${IMPL_OUTPUT_DIRECTORY}/archive/src" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${IMPL_OUTPUT_DIRECTORY}/archive/include" ABSOLUTE)

	set(_BUILTIN_RESOURCES_NAMESPACE_ ${IMPL_NAMESPACE})
	set(_LINK_MODE_ STATIC)

	get_filename_component(BUILTIN_ARCHIVE_INPUT_ABS_ENTRY "${IMPL_INPUT_DIRECTORY}" ABSOLUTE)
	set(BUILTIN_KEY_ENTRY_ABS "${BUILTIN_ARCHIVE_INPUT_ABS_ENTRY}/${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}")

	unset(NBL_RESOURCES_TO_EMBED)
	foreach(IT ${IMPL_BUILTINS})
		if(NBL_LOG_VERBOSE)
			message(STATUS "[${IMPL_TARGET}'s Builtins]: Registered \"${IT}\" key")
		endif()

		LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED ${IT})
	endforeach()

	ADD_CUSTOM_BUILTIN_RESOURCES(${IMPL_TARGET} NBL_RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}" "${_BUILTIN_RESOURCES_NAMESPACE_}" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}" "${_LINK_MODE_}")

	if(IMPL_LINK_TO)
		LINK_BUILTIN_RESOURCES_TO_TARGET(${IMPL_LINK_TO} ${IMPL_TARGET})
	endif()
endfunction()