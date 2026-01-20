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
option(NSC_USE_DEPFILE "Generate depfiles for NSC custom commands" ON)
option(NBL_NSC_DISABLE_CUSTOM_COMMANDS "Disable NSC custom commands" OFF)
option(NBL_NSC_VERBOSE "Enable NSC verbose logging to .log" ON)
option(NSC_SHADER_CACHE "Enable NSC shader cache" ON)
option(NSC_PREPROCESS_CACHE "Enable NSC preprocess cache" ON)
option(NSC_PREPROCESS_PREAMBLE "Enable NSC preprocess preamble" ON)
option(NSC_STDOUT_LOG "Mirror NSC log to stdout" OFF)
set(NSC_CACHE_DIR "" CACHE PATH "Optional root directory for NSC cache files (shader/preprocess)")

function(NBL_CREATE_NSC_COMPILE_RULES)
    set(COMMENT "this code has been autogenerated with Nabla CMake NBL_CREATE_HLSL_COMPILE_RULES utility")
    set(DEVICE_CONFIG_VIEW
[=[

// -> @COMMENT@!
#ifndef _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
#define _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
struct DeviceConfigCaps
{
@CAPS_EVAL@
};
// <- @COMMENT@!
#endif // _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_

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
		-fspv-target-env=vulkan1.3 
		-Wshadow 
		-Wconversion 
		-Wno-local-type-template-args 
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

	list(APPEND REQUIRED_OPTIONS
		-I "${NBL_ROOT_PATH}/include"
		-I "${NBL_ROOT_PATH}/3rdparty/dxc/dxc/external/SPIRV-Headers/include"
		-I "${NBL_ROOT_PATH}/3rdparty/boost/superproject/libs/preprocessor/include"
		-I "${NBL_ROOT_PATH_BINARY}/src/nbl/device/include"
	)
	if(NOT NBL_EMBED_BUILTIN_RESOURCES)
		list(APPEND REQUIRED_OPTIONS -no-nbl-builtins)
	endif()

    set(REQUIRED_SINGLE_ARGS TARGET BINARY_DIR OUTPUT_VAR INPUTS INCLUDE NAMESPACE MOUNT_POINT_DEFINE)
    set(OPTIONAL_SINGLE_ARGS GLOB_DIR)
    cmake_parse_arguments(IMPL "DISCARD_DEFAULT_GLOB;DISABLE_CUSTOM_COMMANDS" "${REQUIRED_SINGLE_ARGS};${OPTIONAL_SINGLE_ARGS};LINK_TO" "COMMON_OPTIONS;DEPENDS" ${ARGV})
    NBL_PARSE_REQUIRED(IMPL ${REQUIRED_SINGLE_ARGS})

	set(_NBL_DISABLE_CUSTOM_COMMANDS FALSE)
	if(NBL_NSC_DISABLE_CUSTOM_COMMANDS OR IMPL_DISABLE_CUSTOM_COMMANDS)
		set(_NBL_DISABLE_CUSTOM_COMMANDS TRUE)
	endif()

	set(IMPL_HLSL_GLOB "")
	if(NOT IMPL_DISCARD_DEFAULT_GLOB)
		set(GLOB_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
		if(IMPL_GLOB_DIR)
			set(GLOB_ROOT "${IMPL_GLOB_DIR}")
		endif()
		get_filename_component(GLOB_ROOT "${GLOB_ROOT}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
		file(GLOB_RECURSE IMPL_HLSL_GLOB CONFIGURE_DEPENDS "${GLOB_ROOT}/*.hlsl")
	endif()

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
		set(NBL_HEADER_GUARD_RAW "${IMPL_TARGET}_${IMPL_NAMESPACE}_SPIRV_KEYS_HPP_INCLUDED")
		string(SHA1 NBL_HEADER_GUARD_HASH "${NBL_HEADER_GUARD_RAW}")
		string(TOUPPER "${NBL_HEADER_GUARD_HASH}" NBL_HEADER_GUARD_HASH_UPPER)
		set(NBL_HEADER_GUARD "SPIRV_KEYS_${NBL_HEADER_GUARD_HASH_UPPER}_HPP_INCLUDED")
		set(INCLUDE_CONTENT_TEMPLATE [=[
#ifndef @NBL_HEADER_GUARD@
#define @NBL_HEADER_GUARD@
$<TARGET_PROPERTY:@IMPL_TARGET@,NBL_HEADER_CONTENT>
#endif
]=])
		string(CONFIGURE "${INCLUDE_CONTENT_TEMPLATE}" INCLUDE_CONTENT @ONLY)

		file(GENERATE OUTPUT ${INCLUDE_FILE}
			CONTENT $<GENEX_EVAL:${INCLUDE_CONTENT}>
			TARGET ${IMPL_TARGET}
		)

		target_sources(${IMPL_TARGET} PUBLIC ${INCLUDE_FILE})
		set_source_files_properties(${INCLUDE_FILE} PROPERTIES 
			HEADER_FILE_ONLY ON
			GENERATED TRUE
		)

		target_compile_definitions(${IMPL_TARGET} INTERFACE $<TARGET_PROPERTY:${IMPL_TARGET},NBL_MOUNT_POINT_DEFINES>)
		target_include_directories(${IMPL_TARGET} INTERFACE ${INCLUDE_DIR})
		set_target_properties(${IMPL_TARGET} PROPERTIES NBL_HEADER_GENERATED_RULE ON)

		set(HEADER_ITEM_VIEW [=[
#include <cstdint>
#include <string>
#include "nbl/core/string/SpirvKeyHelpers.h"

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
	template<nbl::core::StringLiteral Key, typename... Args>
	requires ((... && !std::is_pointer_v<std::remove_cvref_t<Args>>))
	inline constexpr typename nbl::core::detail::StringLiteralBufferType<Key>::type get_spirv_key(const Args&... args)
	{
		return nbl::core::detail::SpirvKeyBuilder<Key>::build(args...);
	}

	template<nbl::core::StringLiteral Key, class Device, typename... Args>
	inline std::string get_spirv_key(const Device* device, const Args&... args)
	{
		const auto key = nbl::core::detail::SpirvKeyBuilder<Key>::build_from_device(device, args...);
		return std::string(key.view());
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

	set(RTE "NSC Rules")
	set(IN "${RTE}/In")
	set(OUT "${RTE}/Out")

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

		macro(NBL_NSC_RESOLVE_CAP_KIND _CAP_KIND_RAW _CAP_STRUCT _CAP_NAME _OUT_KIND)
			set(_CAP_KIND_RAW "${_CAP_KIND_RAW}")
			set(_CAP_STRUCT "${_CAP_STRUCT}")

			if(_CAP_KIND_RAW STREQUAL "custom")
				if(_CAP_STRUCT STREQUAL "")
					ERROR_WHILE_PARSING_ITEM(
						"CAPS entry with kind \"custom\" requires \"struct\".\n"
					)
				endif()
				set(${_OUT_KIND} "${_CAP_STRUCT}")
			else()
				set(${_OUT_KIND} "${_CAP_KIND_RAW}")
			endif()

			if(NOT "${${_OUT_KIND}}" MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
				ERROR_WHILE_PARSING_ITEM(
					"Invalid CAP kind \"${${_OUT_KIND}}\" for ${_CAP_NAME}\n"
					"CAP kinds must be valid C/C++ identifiers."
				)
			endif()
		endmacro()

		macro(NBL_REQUIRE_PYTHON)
			if(NOT Python3_EXECUTABLE)
				find_package(Python3 COMPONENTS Interpreter REQUIRED)
			endif()
		endmacro()

		macro(NBL_NORMALIZE_FLOAT_LITERAL _CAP_NAME _VALUE _MANTISSA_DIGITS _TYPE_LABEL _OUT_VAR)
			NBL_REQUIRE_PYTHON()
			set(_NBL_RAW "${_VALUE}")
			if(_TYPE_LABEL STREQUAL "float")
				if("${_NBL_RAW}" MATCHES "^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?[fF]$")
					string(REGEX REPLACE "[fF]$" "" _NBL_RAW "${_NBL_RAW}")
				endif()
			elseif(_TYPE_LABEL STREQUAL "double")
				if("${_NBL_RAW}" MATCHES "^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?[dD]$")
					string(REGEX REPLACE "[dD]$" "" _NBL_RAW "${_NBL_RAW}")
				endif()
			endif()

			set(_NBL_CANON_DONE FALSE)
			if("${_NBL_RAW}" MATCHES "^[+-]?[0-9]\\.([0-9]+)e([+-][0-9]+)$")
				set(_NBL_MANTISSA "${CMAKE_MATCH_1}")
				set(_NBL_EXPONENT "${CMAKE_MATCH_2}")
				string(LENGTH "${_NBL_MANTISSA}" _NBL_MANTISSA_LEN)
				string(LENGTH "${_NBL_EXPONENT}" _NBL_EXPONENT_LEN)
				math(EXPR _NBL_EXPONENT_DIGITS "${_NBL_EXPONENT_LEN} - 1")
				if(_NBL_MANTISSA_LEN EQUAL ${_MANTISSA_DIGITS} AND _NBL_EXPONENT_DIGITS GREATER_EQUAL 2 AND _NBL_EXPONENT_DIGITS LESS_EQUAL 3)
					string(TOLOWER "${_NBL_RAW}" _NBL_CANON)
					set(_NBL_CANON_DONE TRUE)
				endif()
			endif()

			if(NOT _NBL_CANON_DONE)
				set(_NBL_PY_SCRIPT [=[
import sys,math,struct
t=sys.argv[1]
s=sys.argv[2]
if t=="float" and s[-1:] in ("f","F"):
    s=s[:-1]
if t=="double" and s[-1:] in ("d","D"):
    s=s[:-1]
try:
    x=float(s)
except Exception:
    sys.exit(2)
if t=="float":
    x=struct.unpack("!f",struct.pack("!f",x))[0]
if not math.isfinite(x):
    sys.exit(2)
p=8 if t=="float" else 16
sign="-" if x<0 else ""
x=abs(x)
if x==0.0:
    sys.stdout.write(sign+"0."+"0"*p+"e+00")
    sys.exit(0)
m=x
e=0
while m>=10.0:
    m/=10.0
    e+=1
while m<1.0:
    m*=10.0
    e-=1
digits=[0]*(p+1)
digits[0]=int(m)
frac=m-digits[0]
for i in range(1,p+1):
    frac*=10.0
    d=int(frac)
    if d>9:
        d=9
    digits[i]=d
    frac-=d
frac*=10.0
rd=int(frac)
if rd>9:
    rd=9
rem=frac-rd
ru = rd>5 or (rd==5 and (rem>0 or (digits[p]%2)))
if ru:
    i=p
    while i>=0 and digits[i]==9:
        digits[i]=0
        i-=1
    if i>=0:
        digits[i]+=1
    else:
        digits[0]=1
        for j in range(1,p+1):
            digits[j]=0
        e+=1
es="-" if e<0 else "+"
if e<0:
    e=-e
ew=3 if e>=100 else 2
sys.stdout.write(sign+str(digits[0])+"."+("".join(str(d) for d in digits[1:]))+"e"+es+str(e).zfill(ew))
]=])
				execute_process(
					COMMAND "${Python3_EXECUTABLE}" -c "${_NBL_PY_SCRIPT}" "${_TYPE_LABEL}" "${_NBL_RAW}"
					RESULT_VARIABLE _NBL_FMT_RESULT
					OUTPUT_VARIABLE _NBL_CANON
					OUTPUT_STRIP_TRAILING_WHITESPACE
				)
				if(NOT _NBL_FMT_RESULT EQUAL 0)
					ERROR_WHILE_PARSING_ITEM(
						"Invalid CAP value \"${_VALUE}\" for ${_CAP_NAME}\n"
						"${_TYPE_LABEL} values must be numbers or numeric strings."
					)
				endif()
			endif()
			set(${_OUT_VAR} "${_NBL_CANON}")
		endmacro()

		macro(NBL_HASH_SPIRV_KEY _VALUE _OUT_VAR)
			NBL_REQUIRE_PYTHON()
			set(_NBL_PY_HASH [=[
import sys
s=sys.argv[1]
h=14695981039346656037
for b in s.encode("utf-8"):
    h^=b
    h=(h*1099511628211)&0xFFFFFFFFFFFFFFFF
sys.stdout.write(str(h))
]=])
			execute_process(
				COMMAND "${Python3_EXECUTABLE}" -c "${_NBL_PY_HASH}" "${_VALUE}"
				RESULT_VARIABLE _NBL_HASH_RESULT
				OUTPUT_VARIABLE _NBL_HASH_OUT
				OUTPUT_STRIP_TRAILING_WHITESPACE
			)
			if(NOT _NBL_HASH_RESULT EQUAL 0)
				message(FATAL_ERROR "Failed to hash SPIR-V key \"${_VALUE}\"")
			endif()
			set(${_OUT_VAR} "${_NBL_HASH_OUT}")
		endmacro()

        set(CAP_NAMES "")
        set(CAP_TYPES "")
		set(CAP_KINDS "")
		set(CAP_VALUES_INDEX 0)
        if(HAS_CAPS)
            math(EXPR LAST_CAP "${CAPS_LENGTH} - 1")
            foreach(CAP_IDX RANGE 0 ${LAST_CAP})
				string(JSON MEMBERS_TYPE ERROR_VARIABLE MEMBERS_ERROR TYPE "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members)
				if(MEMBERS_TYPE STREQUAL "ARRAY")
					string(JSON CAP_KIND_RAW ERROR_VARIABLE CAP_KIND_ERROR GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} kind)
					if(CAP_KIND_ERROR)
						set(CAP_KIND_RAW limits)
					endif()

					string(JSON CAP_STRUCT ERROR_VARIABLE CAP_STRUCT_ERROR GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} struct)
					if(CAP_STRUCT_ERROR)
						set(CAP_STRUCT "")
					endif()

					NBL_NSC_RESOLVE_CAP_KIND("${CAP_KIND_RAW}" "${CAP_STRUCT}" "member group" CAP_KIND)

					string(JSON MEMBERS_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members)
					if(MEMBERS_LENGTH GREATER 0)
						math(EXPR LAST_MEMBER "${MEMBERS_LENGTH} - 1")
						foreach(MEMBER_IDX RANGE 0 ${LAST_MEMBER})
							string(JSON CAP_NAME GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members ${MEMBER_IDX} name)
							string(JSON CAP_TYPE GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members ${MEMBER_IDX} type)

							if(NOT CAP_TYPE MATCHES "^(bool|uint16_t|uint32_t|uint64_t|int16_t|int32_t|int64_t|float|double)$")
								ERROR_WHILE_PARSING_ITEM(
									"Invalid CAP type \"${CAP_TYPE}\" for ${CAP_NAME}\n"
									"Allowed types are: bool, uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double"
								)
							endif()

							string(JSON CAP_VALUES_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members ${MEMBER_IDX} values)

							set(VALUES "")
							math(EXPR LAST_VAL "${CAP_VALUES_LENGTH} - 1")
							foreach(VAL_IDX RANGE 0 ${LAST_VAL})
								string(JSON VALUE GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members ${MEMBER_IDX} values ${VAL_IDX})
								string(JSON VAL_TYPE TYPE "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} members ${MEMBER_IDX} values ${VAL_IDX})

								if(CAP_TYPE STREQUAL "float")
									if(NOT (VAL_TYPE STREQUAL "STRING" OR VAL_TYPE STREQUAL "NUMBER"))
										ERROR_WHILE_PARSING_ITEM(
											"Invalid CAP value \"${VALUE}\" for ${CAP_NAME}\n"
											"Float values must be numbers or numeric strings."
										)
									endif()
									NBL_NORMALIZE_FLOAT_LITERAL("${CAP_NAME}" "${VALUE}" 8 "float" VALUE)
								elseif(CAP_TYPE STREQUAL "double")
									if(NOT (VAL_TYPE STREQUAL "STRING" OR VAL_TYPE STREQUAL "NUMBER"))
										ERROR_WHILE_PARSING_ITEM(
											"Invalid CAP value \"${VALUE}\" for ${CAP_NAME}\n"
											"Double values must be numbers or numeric strings."
										)
									endif()
									NBL_NORMALIZE_FLOAT_LITERAL("${CAP_NAME}" "${VALUE}" 16 "double" VALUE)
								elseif(NOT VAL_TYPE STREQUAL "NUMBER")
									ERROR_WHILE_PARSING_ITEM(
										"Invalid CAP value \"${VALUE}\" for CAP \"${CAP_NAME}\" of type ${CAP_TYPE}\n"
										"Use numbers for uint*_t and 0/1 for bools."
									)
								elseif(NOT VAL_TYPE STREQUAL "NUMBER")
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

							set(CAP_VALUES_${CAP_VALUES_INDEX} "${VALUES}")
							list(APPEND CAP_NAMES "${CAP_NAME}")
							list(APPEND CAP_TYPES "${CAP_TYPE}")
							list(APPEND CAP_KINDS "${CAP_KIND}")
							math(EXPR CAP_VALUES_INDEX "${CAP_VALUES_INDEX} + 1")
						endforeach()
					endif()
				else()
					if(NOT MEMBERS_ERROR)
						ERROR_WHILE_PARSING_ITEM(
							"CAPS.members must be an array when provided."
						)
					endif()

					string(JSON CAP_KIND_RAW ERROR_VARIABLE CAP_KIND_ERROR GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} kind)
					string(JSON CAP_NAME GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} name)
					string(JSON CAP_TYPE GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} type)

					if(CAP_KIND_ERROR)
						set(CAP_KIND_RAW limits) # I assume its limit by default (or when invalid value present, currently)
					endif()

					string(JSON CAP_STRUCT ERROR_VARIABLE CAP_STRUCT_ERROR GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} struct)
					if(CAP_STRUCT_ERROR)
						set(CAP_STRUCT "")
					endif()

					NBL_NSC_RESOLVE_CAP_KIND("${CAP_KIND_RAW}" "${CAP_STRUCT}" "${CAP_NAME}" CAP_KIND)

					if(NOT CAP_TYPE MATCHES "^(bool|uint16_t|uint32_t|uint64_t|int16_t|int32_t|int64_t|float|double)$")
						ERROR_WHILE_PARSING_ITEM(
							"Invalid CAP type \"${CAP_TYPE}\" for ${CAP_NAME}\n"
							"Allowed types are: bool, uint16_t, uint32_t, uint64_t, int16_t, int32_t, int64_t, float, double"
						)
					endif()

					string(JSON CAP_VALUES_LENGTH LENGTH "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} values)

					set(VALUES "")
					math(EXPR LAST_VAL "${CAP_VALUES_LENGTH} - 1")
					foreach(VAL_IDX RANGE 0 ${LAST_VAL})
						string(JSON VALUE GET "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} values ${VAL_IDX})
						string(JSON VAL_TYPE TYPE "${IMPL_INPUTS}" ${INDEX} CAPS ${CAP_IDX} values ${VAL_IDX})

						if(CAP_TYPE STREQUAL "float")
							if(NOT (VAL_TYPE STREQUAL "STRING" OR VAL_TYPE STREQUAL "NUMBER"))
								ERROR_WHILE_PARSING_ITEM(
									"Invalid CAP value \"${VALUE}\" for ${CAP_NAME}\n"
									"Float values must be numbers or numeric strings."
								)
							endif()
							NBL_NORMALIZE_FLOAT_LITERAL("${CAP_NAME}" "${VALUE}" 8 "float" VALUE)
						elseif(CAP_TYPE STREQUAL "double")
							if(NOT (VAL_TYPE STREQUAL "STRING" OR VAL_TYPE STREQUAL "NUMBER"))
								ERROR_WHILE_PARSING_ITEM(
									"Invalid CAP value \"${VALUE}\" for ${CAP_NAME}\n"
									"Double values must be numbers or numeric strings."
								)
							endif()
							NBL_NORMALIZE_FLOAT_LITERAL("${CAP_NAME}" "${VALUE}" 16 "double" VALUE)
						elseif(NOT VAL_TYPE STREQUAL "NUMBER")
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

					set(CAP_VALUES_${CAP_VALUES_INDEX} "${VALUES}")
					list(APPEND CAP_NAMES "${CAP_NAME}")
					list(APPEND CAP_TYPES "${CAP_TYPE}")
					list(APPEND CAP_KINDS "${CAP_KIND}")
					math(EXPR CAP_VALUES_INDEX "${CAP_VALUES_INDEX} + 1")
				endif()
            endforeach()
        endif()

        list(LENGTH CAP_NAMES NUM_CAPS)

		set(TARGET_INPUT "${INPUT}")
		if(NOT IS_ABSOLUTE "${TARGET_INPUT}")
			set(TARGET_INPUT "${CMAKE_CURRENT_SOURCE_DIR}/${TARGET_INPUT}")
		endif()
		if(IMPL_HLSL_GLOB)
			get_filename_component(_ABS_TARGET_INPUT "${TARGET_INPUT}" ABSOLUTE)
			list(REMOVE_ITEM IMPL_HLSL_GLOB "${TARGET_INPUT}" "${_ABS_TARGET_INPUT}")
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

		if(NUM_CAPS GREATER 0)
			set(KIND_ORDER "")
			foreach(_NBL_KIND IN LISTS CAP_KINDS)
				list(FIND KIND_ORDER "${_NBL_KIND}" _NBL_KIND_INDEX)
				if(_NBL_KIND_INDEX EQUAL -1)
					list(APPEND KIND_ORDER "${_NBL_KIND}")
				endif()
			endforeach()

			set(ORDERED_KINDS "${KIND_ORDER}")

			foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
				unset(_NBL_KIND_INDICES_${_NBL_KIND})
			endforeach()

			math(EXPR LAST_CAP "${NUM_CAPS} - 1")
			foreach(i RANGE 0 ${LAST_CAP})
				list(GET CAP_KINDS ${i} _NBL_KIND)
				set(_NBL_ORIG_CAP_VALUES_${i} "${CAP_VALUES_${i}}")
				list(APPEND _NBL_KIND_INDICES_${_NBL_KIND} ${i})
			endforeach()

			set(_NBL_ORDERED_INDICES "")
			foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
				if(DEFINED _NBL_KIND_INDICES_${_NBL_KIND})
					list(APPEND _NBL_ORDERED_INDICES ${_NBL_KIND_INDICES_${_NBL_KIND}})
				endif()
			endforeach()

			set(_NBL_ORDERED_CAP_NAMES "")
			set(_NBL_ORDERED_CAP_TYPES "")
			set(_NBL_ORDERED_CAP_KINDS "")
			set(_NBL_ORDERED_VALUES_INDEX 0)
			foreach(_NBL_INDEX IN LISTS _NBL_ORDERED_INDICES)
				list(GET CAP_NAMES ${_NBL_INDEX} _NBL_CAP_NAME)
				list(GET CAP_TYPES ${_NBL_INDEX} _NBL_CAP_TYPE)
				list(GET CAP_KINDS ${_NBL_INDEX} _NBL_CAP_KIND)
				set(_NBL_CAP_VALUES "${_NBL_ORIG_CAP_VALUES_${_NBL_INDEX}}")
				list(APPEND _NBL_ORDERED_CAP_NAMES "${_NBL_CAP_NAME}")
				list(APPEND _NBL_ORDERED_CAP_TYPES "${_NBL_CAP_TYPE}")
				list(APPEND _NBL_ORDERED_CAP_KINDS "${_NBL_CAP_KIND}")
				set(CAP_VALUES_${_NBL_ORDERED_VALUES_INDEX} "${_NBL_CAP_VALUES}")
				math(EXPR _NBL_ORDERED_VALUES_INDEX "${_NBL_ORDERED_VALUES_INDEX} + 1")
			endforeach()

			set(CAP_NAMES "${_NBL_ORDERED_CAP_NAMES}")
			set(CAP_TYPES "${_NBL_ORDERED_CAP_TYPES}")
			set(CAP_KINDS "${_NBL_ORDERED_CAP_KINDS}")
			list(LENGTH CAP_NAMES NUM_CAPS)
		else()
			set(ORDERED_KINDS "")
		endif()

		list(LENGTH ORDERED_KINDS ORDERED_KIND_COUNT)
		set(NON_DEVICE_KINDS "")
		set(HAS_LIMITS FALSE)
		set(HAS_FEATURES FALSE)
		foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
			if(_NBL_KIND STREQUAL "limits")
				set(HAS_LIMITS TRUE)
			elseif(_NBL_KIND STREQUAL "features")
				set(HAS_FEATURES TRUE)
			else()
				list(APPEND NON_DEVICE_KINDS "${_NBL_KIND}")
			endif()
		endforeach()
		list(LENGTH NON_DEVICE_KINDS NON_DEVICE_COUNT)

		string(MAKE_C_IDENTIFIER "${BASE_KEY}" BASE_KEY_IDENT)
		string(MD5 BASE_KEY_HASH "${BASE_KEY}")
		string(SUBSTRING "${BASE_KEY_HASH}" 0 8 BASE_KEY_HASH8)
		set(KIND_PREFIX "${BASE_KEY_IDENT}_${BASE_KEY_HASH8}")

		set(MATCH_KINDS "")
		foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
			list(APPEND MATCH_KINDS "${_NBL_KIND}")
		endforeach()

		foreach(_NBL_KIND IN LISTS MATCH_KINDS)
			set(_NBL_KIND_MEMBERS_${_NBL_KIND} "")
			set(_NBL_KIND_TYPES_${_NBL_KIND} "")
		endforeach()

		if(NUM_CAPS GREATER 0)
			math(EXPR _NBL_LAST_CAP "${NUM_CAPS} - 1")
			foreach(i RANGE ${_NBL_LAST_CAP})
				list(GET CAP_KINDS ${i} _NBL_KIND)
				list(GET CAP_NAMES ${i} _NBL_CAP)
				list(GET CAP_TYPES ${i} _NBL_TYPE)
				list(FIND _NBL_KIND_MEMBERS_${_NBL_KIND} "${_NBL_CAP}" _NBL_MEMBER_INDEX)
				if(_NBL_MEMBER_INDEX EQUAL -1)
					list(APPEND _NBL_KIND_MEMBERS_${_NBL_KIND} "${_NBL_CAP}")
					list(APPEND _NBL_KIND_TYPES_${_NBL_KIND} "${_NBL_TYPE}")
				endif()
			endforeach()
		endif()

		list(LENGTH CAP_NAMES CAP_COUNT)

		set(RETVAL_FMT "${BASE_KEY}")
		set(RETVAL_ARGS "")
		set(CX_CAPACITY 0)
		string(LENGTH "${BASE_KEY}" CX_BASE_LEN)
		math(EXPR CX_CAPACITY "${CX_BASE_LEN} + 4 + 24")
		if(CAP_COUNT GREATER 0)
			math(EXPR LAST_CAP "${CAP_COUNT} - 1")
			set(PREV_KIND "")
			foreach(i RANGE ${LAST_CAP})
				list(GET CAP_NAMES ${i} CAP)
				list(GET CAP_KINDS ${i} KIND)
				list(GET CAP_TYPES ${i} TYPE)
				if(NOT KIND STREQUAL PREV_KIND)
					string(APPEND RETVAL_FMT "__${KIND}")
					string(LENGTH "${KIND}" KIND_LEN)
					math(EXPR CX_CAPACITY "${CX_CAPACITY} + 2 + ${KIND_LEN}")
					set(PREV_KIND "${KIND}")
				endif()
				string(APPEND RETVAL_FMT ".${CAP}_%s")
				list(APPEND RETVAL_ARGS "nbl_spirv_${KIND}.${CAP}")
				string(LENGTH "${CAP}" CAP_LEN)
				math(EXPR CX_CAPACITY "${CX_CAPACITY} + 2 + ${CAP_LEN}")
				if(TYPE STREQUAL "bool")
					set(DIGITS 1)
				elseif(TYPE STREQUAL "uint16_t")
					set(DIGITS 5)
				elseif(TYPE STREQUAL "uint32_t")
					set(DIGITS 10)
				elseif(TYPE STREQUAL "int16_t")
					set(DIGITS 6)
				elseif(TYPE STREQUAL "int32_t")
					set(DIGITS 11)
				elseif(TYPE STREQUAL "int64_t")
					set(DIGITS 20)
				elseif(TYPE STREQUAL "uint64_t")
					set(DIGITS 20)
				elseif(TYPE STREQUAL "float")
					set(DIGITS 16)
				elseif(TYPE STREQUAL "double")
					set(DIGITS 24)
				else()
					set(DIGITS 20)
				endif()
				math(EXPR CX_CAPACITY "${CX_CAPACITY} + ${DIGITS}")
			endforeach()
		endif()
		string(APPEND RETVAL_FMT ".spv")
		if(RETVAL_ARGS)
			string(JOIN ", " RETVAL_ARGS_JOINED ${RETVAL_ARGS})
			set(RETVAL_ARGS_STR ", ${RETVAL_ARGS_JOINED}")
		else()
			set(RETVAL_ARGS_STR "")
		endif()
		string(CONFIGURE [=[
		typename StringLiteralBufferType<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>::type nbl_spirv_full = {};
		nbl::core::detail::append_printf_s<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@RETVAL_FMT@")>(nbl_spirv_full@RETVAL_ARGS_STR@);
		retval.append("$<CONFIG>/");
		nbl::core::detail::put(retval, nbl::core::FNV1a_64(nbl_spirv_full.view()));
		retval.append(".spv");
]=] 		RETVAL_EVAL_CONSTEXPR @ONLY)

		set(SPIRV_CUSTOM_TRAITS "")
		foreach(_NBL_KIND IN LISTS MATCH_KINDS)
			set(_NBL_MEMBER_LINES "")
			list(LENGTH _NBL_KIND_MEMBERS_${_NBL_KIND} _NBL_MEMBER_COUNT)
			set(KIND_TRAIT "${KIND_PREFIX}_${_NBL_KIND}")
			if(_NBL_MEMBER_COUNT GREATER 0)
				math(EXPR _NBL_MEMBER_LAST "${_NBL_MEMBER_COUNT} - 1")
				foreach(_NBL_MEMBER_INDEX RANGE ${_NBL_MEMBER_LAST})
					list(GET _NBL_KIND_MEMBERS_${_NBL_KIND} ${_NBL_MEMBER_INDEX} _NBL_MEMBER_NAME)
					list(GET _NBL_KIND_TYPES_${_NBL_KIND} ${_NBL_MEMBER_INDEX} _NBL_MEMBER_TYPE)
					set(MEMBER_NAME "${_NBL_MEMBER_NAME}")
					set(MEMBER_TYPE "${_NBL_MEMBER_TYPE}")
					string(CONFIGURE [=[
			requires std::is_same_v<std::remove_cvref_t<decltype(v.@MEMBER_NAME@)>, @MEMBER_TYPE@>;
]=] 					_NBL_MEMBER_LINE @ONLY)
					string(APPEND _NBL_MEMBER_LINES "${_NBL_MEMBER_LINE}")
				endforeach()
				set(KIND "${KIND_TRAIT}")
				set(MEMBER_LINES "${_NBL_MEMBER_LINES}")
				string(CONFIGURE [=[
	template<class T>
	struct SpirvPerm_@KIND@
	{
		static constexpr bool value = requires(const T& v)
		{
@MEMBER_LINES@		};
	};

]=] 			_NBL_CUSTOM_TRAIT @ONLY)
			else()
				set(KIND "${KIND_TRAIT}")
				string(CONFIGURE [=[
	template<class T>
	struct SpirvPerm_@KIND@
	{
		static constexpr bool value = false;
	};

]=] 			_NBL_CUSTOM_TRAIT @ONLY)
			endif()
			string(APPEND SPIRV_CUSTOM_TRAITS "${_NBL_CUSTOM_TRAIT}")
		endforeach()

		set(SPIRV_BUILD_REQUIRES "")
		if(ORDERED_KIND_COUNT EQUAL 0)
			set(SPIRV_BUILD_REQUIRES "requires (sizeof...(Args) == 0)")
		else()
			set(_NBL_REQS "")
			set(_NBL_KIND_INDEX 0)
			foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
				set(KIND_TRAIT "${KIND_PREFIX}_${_NBL_KIND}")
				list(APPEND _NBL_REQS "SpirvPerm_${KIND_TRAIT}<std::remove_cvref_t<std::tuple_element_t<${_NBL_KIND_INDEX}, std::tuple<Args...>>>>::value")
				math(EXPR _NBL_KIND_INDEX "${_NBL_KIND_INDEX} + 1")
			endforeach()
			string(JOIN " && " _NBL_REQS_JOINED ${_NBL_REQS})
			set(SPIRV_BUILD_REQUIRES "requires (sizeof...(Args) == ${ORDERED_KIND_COUNT} && ${_NBL_REQS_JOINED})")
		endif()

		set(SPIRV_ARG_DECLS "")
		set(_NBL_KIND_INDEX 0)
		foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
			string(APPEND SPIRV_ARG_DECLS "\t\tconst auto& nbl_spirv_${_NBL_KIND} = std::get<${_NBL_KIND_INDEX}>(std::forward_as_tuple(args...));\n")
			math(EXPR _NBL_KIND_INDEX "${_NBL_KIND_INDEX} + 1")
		endforeach()

		set(SPIRV_BUILD_FROM_DEVICE_REQUIRES "")
		set(_NBL_DEVICE_REQS "")
		if(HAS_LIMITS)
			list(APPEND _NBL_DEVICE_REQS "nbl::core::detail::spirv_device_has_limits<Device>")
		endif()
		if(HAS_FEATURES)
			list(APPEND _NBL_DEVICE_REQS "nbl::core::detail::spirv_device_has_features<Device>")
		endif()
		if(NON_DEVICE_COUNT EQUAL 0)
			list(APPEND _NBL_DEVICE_REQS "sizeof...(Args) == 0")
		else()
			list(APPEND _NBL_DEVICE_REQS "sizeof...(Args) == ${NON_DEVICE_COUNT}")
			set(_NBL_REQS "")
			set(_NBL_KIND_INDEX 0)
			foreach(_NBL_KIND IN LISTS NON_DEVICE_KINDS)
				set(KIND_TRAIT "${KIND_PREFIX}_${_NBL_KIND}")
				list(APPEND _NBL_REQS "SpirvPerm_${KIND_TRAIT}<std::remove_cvref_t<std::tuple_element_t<${_NBL_KIND_INDEX}, std::tuple<Args...>>>>::value")
				math(EXPR _NBL_KIND_INDEX "${_NBL_KIND_INDEX} + 1")
			endforeach()
			if(_NBL_REQS)
				string(JOIN " && " _NBL_REQS_JOINED ${_NBL_REQS})
				list(APPEND _NBL_DEVICE_REQS "${_NBL_REQS_JOINED}")
			endif()
		endif()
		string(JOIN " && " SPIRV_DEVICE_REQUIRES_EXPR ${_NBL_DEVICE_REQS})
		set(SPIRV_BUILD_FROM_DEVICE_REQUIRES "requires (${SPIRV_DEVICE_REQUIRES_EXPR})")

		set(SPIRV_BUILD_FROM_DEVICE_ARGS "")
		set(_NBL_ARG_INDEX 0)
		foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
			if(_NBL_KIND STREQUAL "limits")
				list(APPEND SPIRV_BUILD_FROM_DEVICE_ARGS "nbl::core::detail::spirv_device_get_limits(device)")
			elseif(_NBL_KIND STREQUAL "features")
				list(APPEND SPIRV_BUILD_FROM_DEVICE_ARGS "nbl::core::detail::spirv_device_get_features(device)")
			else()
				list(APPEND SPIRV_BUILD_FROM_DEVICE_ARGS "std::get<${_NBL_ARG_INDEX}>(std::forward_as_tuple(args...))")
				math(EXPR _NBL_ARG_INDEX "${_NBL_ARG_INDEX} + 1")
			endif()
		endforeach()
		if(SPIRV_BUILD_FROM_DEVICE_ARGS)
			string(JOIN ", " SPIRV_BUILD_FROM_DEVICE_ARGS_JOINED ${SPIRV_BUILD_FROM_DEVICE_ARGS})
		else()
			set(SPIRV_BUILD_FROM_DEVICE_ARGS_JOINED "")
		endif()

		set(SPIRV_TRIVIAL_ASSERTS "")

		set(HEADER_ITEM_VIEW [=[
namespace nbl::core::detail {
	template<>
	struct StringLiteralBufferType<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>
	{
		using type = StringLiteralBuffer<@CX_CAPACITY@ + 1>;
	};

	template<>
	struct SpirvKeyBuilder<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>
	{
@SPIRV_CUSTOM_TRAITS@		template<typename... Args>
		@SPIRV_BUILD_REQUIRES@
		static constexpr typename StringLiteralBufferType<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>::type build(const Args&... args)
		{
@SPIRV_ARG_DECLS@@SPIRV_TRIVIAL_ASSERTS@			typename StringLiteralBufferType<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>::type retval = {};
@RETVAL_EVAL_CONSTEXPR@
			return retval;
		}

		template<class Device, typename... Args>
		@SPIRV_BUILD_FROM_DEVICE_REQUIRES@
		static constexpr typename StringLiteralBufferType<NBL_CORE_UNIQUE_STRING_LITERAL_TYPE("@BASE_KEY@")>::type build_from_device(const Device* device, const Args&... args)
		{
			return build(@SPIRV_BUILD_FROM_DEVICE_ARGS_JOINED@);
		}
	};
}

]=])
		string(CONFIGURE "${HEADER_ITEM_VIEW}" HEADER_ITEM_EVAL @ONLY)
		set_property(TARGET ${IMPL_TARGET} APPEND_STRING PROPERTY NBL_HEADER_CONTENT "${HEADER_ITEM_EVAL}")

		function(GENERATE_KEYS PREFIX CAP_INDEX)
			set(CAPS_VALUES_PART "${ARGN}")
			if(NUM_CAPS EQUAL 0 OR CAP_INDEX EQUAL ${NUM_CAPS})
				set(FINAL_KEY "${BASE_KEY}${PREFIX}.spv") # always add ext even if its already there to make sure asset loader always is able to load as IShader
				NBL_HASH_SPIRV_KEY("${FINAL_KEY}" FINAL_KEY_HASH)
				set(HASHED_KEY "${FINAL_KEY_HASH}.spv")
				set(CONFIG_FILE_TARGET_OUTPUT "${IMPL_BINARY_DIR}/${FINAL_KEY_HASH}")
				set(CONFIG_FILE "${CONFIG_FILE_TARGET_OUTPUT}.in.hlsl")
				set(CAPS_EVAL "")
				if(NUM_CAPS GREATER 0)
					set(CAPS_EVAL_LIMITS "")
					set(CAPS_EVAL_FEATURES "")
					set(_NBL_CUSTOM_KIND_LIST "")
					foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
						if(NOT _NBL_KIND STREQUAL "limits" AND NOT _NBL_KIND STREQUAL "features")
							list(APPEND _NBL_CUSTOM_KIND_LIST "${_NBL_KIND}")
							set(_NBL_CUSTOM_LINES_${_NBL_KIND} "")
						endif()
					endforeach()

					math(EXPR _NBL_LAST_CAP "${NUM_CAPS} - 1")
					foreach(i RANGE 0 ${_NBL_LAST_CAP})
						list(GET CAP_NAMES ${i} _NBL_CAP_NAME)
						list(GET CAP_TYPES ${i} _NBL_CAP_TYPE)
						list(GET CAP_KINDS ${i} _NBL_CAP_KIND)
						list(GET CAPS_VALUES_PART ${i} _NBL_CAP_VALUE)
						set(MEMBER_NAME "${_NBL_CAP_NAME}")
						set(MEMBER_TYPE "${_NBL_CAP_TYPE}")
						set(MEMBER_VALUE "${_NBL_CAP_VALUE}")
						if(MEMBER_TYPE STREQUAL "double" AND MEMBER_VALUE STREQUAL "1.7976931348623165e+308")
							set(MEMBER_VALUE "1.7976931348623157e+308")
						endif()
						if(MEMBER_TYPE STREQUAL "double")
							set(MEMBER_VALUE "${MEMBER_VALUE}L")
						endif()
						string(CONFIGURE [=[
NBL_CONSTEXPR_STATIC_INLINE @MEMBER_TYPE@ @MEMBER_NAME@ = (@MEMBER_TYPE@) @MEMBER_VALUE@;
]=] _NBL_MEMBER_LINE @ONLY)
						if(_NBL_CAP_KIND STREQUAL "limits")
							string(APPEND CAPS_EVAL_LIMITS "	${_NBL_MEMBER_LINE}")
						elseif(_NBL_CAP_KIND STREQUAL "features")
							string(APPEND CAPS_EVAL_FEATURES "	${_NBL_MEMBER_LINE}")
						else()
							set(_NBL_CUSTOM_LINE_VAR "_NBL_CUSTOM_LINES_${_NBL_CAP_KIND}")
							set(${_NBL_CUSTOM_LINE_VAR} "${${_NBL_CUSTOM_LINE_VAR}}		${_NBL_MEMBER_LINE}")
						endif()
					endforeach()

					if(CAPS_EVAL_LIMITS)
						string(APPEND CAPS_EVAL "	// limits\n")
						string(APPEND CAPS_EVAL "${CAPS_EVAL_LIMITS}")
					endif()
					if(CAPS_EVAL_FEATURES)
						string(APPEND CAPS_EVAL "	// features\n")
						string(APPEND CAPS_EVAL "${CAPS_EVAL_FEATURES}")
					endif()

					set(_NBL_HAS_CUSTOM FALSE)
					foreach(_NBL_KIND IN LISTS _NBL_CUSTOM_KIND_LIST)
						if(_NBL_CUSTOM_LINES_${_NBL_KIND})
							set(_NBL_HAS_CUSTOM TRUE)
						endif()
					endforeach()

					if(_NBL_HAS_CUSTOM)
						string(APPEND CAPS_EVAL "	// custom structs\n")
						foreach(_NBL_KIND IN LISTS ORDERED_KINDS)
							if(NOT _NBL_KIND STREQUAL "limits" AND NOT _NBL_KIND STREQUAL "features")
								if(_NBL_CUSTOM_LINES_${_NBL_KIND})
									set(NBL_KIND_NAME "${_NBL_KIND}")
									set(MEMBER_LINES "${_NBL_CUSTOM_LINES_${_NBL_KIND}}")
									string(CONFIGURE [=[
	struct @NBL_KIND_NAME@
	{
@MEMBER_LINES@	};
]=] _NBL_KIND_STRUCT @ONLY)
									string(APPEND CAPS_EVAL "${_NBL_KIND_STRUCT}")
								endif()
							endif()
						endforeach()
					endif()
				endif()
				if(CAPS_EVAL STREQUAL "")
					set(CAPS_EVAL "	// no caps\n")
				endif()
				string(CONFIGURE "${DEVICE_CONFIG_VIEW}" CONFIG_CONTENT @ONLY)
				set(_NBL_CONFIG_WRITE TRUE)
				if(EXISTS "${CONFIG_FILE}")
					file(READ "${CONFIG_FILE}" _NBL_CONFIG_OLD)
					if(_NBL_CONFIG_OLD STREQUAL "${CONFIG_CONTENT}")
						set(_NBL_CONFIG_WRITE FALSE)
					endif()
				endif()
				if(_NBL_CONFIG_WRITE)
					file(WRITE "${CONFIG_FILE}" "${CONFIG_CONTENT}")
				endif()
				list(APPEND DEPENDS_ON "${TARGET_INPUT}" "${CONFIG_FILE}")

				# generate keys and commands for compiling shaders
				set(FINAL_KEY_REL_PATH "$<CONFIG>/${HASHED_KEY}")
				set(TARGET_OUTPUT "${IMPL_BINARY_DIR}/${FINAL_KEY_REL_PATH}")
				set(DEPFILE_PATH "${TARGET_OUTPUT}.dep")
				set(NBL_NSC_LOG_PATH "${TARGET_OUTPUT}.log")
				set(NBL_NSC_PREPROCESSED_PATH "${TARGET_OUTPUT}.pre.hlsl")
				if(NSC_CACHE_DIR)
					get_filename_component(NBL_NSC_CACHE_ROOT "${NSC_CACHE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_BINARY_DIR}")
					file(RELATIVE_PATH NBL_NSC_CACHE_REL "${IMPL_BINARY_DIR}" "${TARGET_OUTPUT}")
					set(NBL_NSC_CACHE_PATH "${NBL_NSC_CACHE_ROOT}/${NBL_NSC_CACHE_REL}.ppcache")
					set(NBL_NSC_PREPROCESS_CACHE_PATH "${NBL_NSC_CACHE_ROOT}/${NBL_NSC_CACHE_REL}.ppcache.pre")
				else()
					set(NBL_NSC_CACHE_PATH "${TARGET_OUTPUT}.ppcache")
					set(NBL_NSC_PREPROCESS_CACHE_PATH "${TARGET_OUTPUT}.ppcache.pre")
				endif()

				set(NBL_NSC_DEPFILE_ARGS "")
				if(NSC_USE_DEPFILE)
					set(NBL_NSC_DEPFILE_ARGS -MD -MF "${DEPFILE_PATH}")
				endif()

				set(NBL_NSC_CACHE_ARGS "")
				if(NSC_SHADER_CACHE)
					list(APPEND NBL_NSC_CACHE_ARGS -nbl-shader-cache)
					if(NSC_CACHE_DIR)
						list(APPEND NBL_NSC_CACHE_ARGS -shader-cache-file "${NBL_NSC_CACHE_PATH}")
					endif()
				endif()
				if(NSC_PREPROCESS_CACHE)
					list(APPEND NBL_NSC_CACHE_ARGS -nbl-preprocess-cache)
					if(NSC_CACHE_DIR)
						list(APPEND NBL_NSC_CACHE_ARGS -preprocess-cache-file "${NBL_NSC_PREPROCESS_CACHE_PATH}")
					endif()
					if(NSC_PREPROCESS_PREAMBLE)
						list(APPEND NBL_NSC_CACHE_ARGS -nbl-preprocess-preamble)
					endif()
				endif()
				if(NSC_STDOUT_LOG)
					list(APPEND NBL_NSC_CACHE_ARGS -nbl-stdout-log)
				endif()

				set(NBL_NSC_COMPILE_COMMAND
					"$<TARGET_FILE:nsc>"
					-Fc "${TARGET_OUTPUT}"
					${COMPILE_OPTIONS} ${REQUIRED_OPTIONS} ${IMPL_COMMON_OPTIONS}
					${NBL_NSC_DEPFILE_ARGS}
					$<$<BOOL:${NBL_NSC_VERBOSE}>:-verbose>
					${NBL_NSC_CACHE_ARGS}
					-FI "${CONFIG_FILE}"
					"${TARGET_INPUT}"
				)

				get_filename_component(NBL_NSC_INPUT_NAME "${TARGET_INPUT}" NAME)
				get_filename_component(NBL_NSC_CONFIG_NAME "${CONFIG_FILE}" NAME)
				set(NBL_NSC_COMMENT_LEFT "${NBL_NSC_INPUT_NAME}")
				set(NBL_NSC_COMMENT_RIGHT "${NBL_NSC_CONFIG_NAME}")
				if(NBL_NSC_INPUT_NAME MATCHES "\\.in\\.hlsl$")
					set(NBL_NSC_COMMENT_LEFT "${NBL_NSC_CONFIG_NAME}")
					set(NBL_NSC_COMMENT_RIGHT "${NBL_NSC_INPUT_NAME}")
				endif()
				set(NBL_NSC_MAIN_DEPENDENCY "${TARGET_INPUT}")
				if(TARGET nsc)
					if(CMAKE_GENERATOR MATCHES "Visual Studio")
						list(APPEND DEPENDS_ON "$<TARGET_FILE:nsc>")
					else()
						list(APPEND DEPENDS_ON nsc)
					endif()
				endif()
				set(NBL_NSC_BYPRODUCTS "${NBL_NSC_LOG_PATH}")
				if(NSC_USE_DEPFILE)
					list(APPEND NBL_NSC_BYPRODUCTS "${DEPFILE_PATH}")
				endif()
				if(NSC_SHADER_CACHE)
					list(APPEND NBL_NSC_BYPRODUCTS "${NBL_NSC_CACHE_PATH}")
				endif()
				if(NSC_PREPROCESS_CACHE)
					list(APPEND NBL_NSC_BYPRODUCTS "${NBL_NSC_PREPROCESS_CACHE_PATH}")
					list(APPEND NBL_NSC_BYPRODUCTS "${NBL_NSC_PREPROCESSED_PATH}")
				endif()

				set(NBL_NSC_CUSTOM_COMMAND_ARGS
					OUTPUT "${TARGET_OUTPUT}"
					BYPRODUCTS ${NBL_NSC_BYPRODUCTS}
					COMMAND ${NBL_NSC_COMPILE_COMMAND}
					DEPENDS ${DEPENDS_ON}
					COMMENT "${NBL_NSC_COMMENT_LEFT} (${NBL_NSC_COMMENT_RIGHT})"
					VERBATIM
					COMMAND_EXPAND_LISTS
				)
				if(NBL_NSC_MAIN_DEPENDENCY)
					list(APPEND NBL_NSC_CUSTOM_COMMAND_ARGS MAIN_DEPENDENCY "${NBL_NSC_MAIN_DEPENDENCY}")
				endif()
				if(NSC_USE_DEPFILE)
					list(APPEND NBL_NSC_CUSTOM_COMMAND_ARGS DEPFILE "${DEPFILE_PATH}")
				endif()
				if(NOT _NBL_DISABLE_CUSTOM_COMMANDS)
					add_custom_command(${NBL_NSC_CUSTOM_COMMAND_ARGS})
				endif()
				set(NBL_NSC_OUT_FILES "")
				if(NOT _NBL_DISABLE_CUSTOM_COMMANDS)
					set(NBL_NSC_OUT_FILES "${TARGET_OUTPUT}" "${NBL_NSC_LOG_PATH}")
					if(NSC_USE_DEPFILE)
						list(APPEND NBL_NSC_OUT_FILES "${DEPFILE_PATH}")
					endif()
					if(NSC_SHADER_CACHE)
						list(APPEND NBL_NSC_OUT_FILES "${NBL_NSC_CACHE_PATH}")
					endif()
					if(NSC_PREPROCESS_CACHE)
						list(APPEND NBL_NSC_OUT_FILES "${NBL_NSC_PREPROCESS_CACHE_PATH}")
						list(APPEND NBL_NSC_OUT_FILES "${NBL_NSC_PREPROCESSED_PATH}")
					endif()
					set_source_files_properties(${NBL_NSC_OUT_FILES} PROPERTIES GENERATED TRUE)
				endif()

				set(HEADER_ONLY_LIKE "")
				set(ADD_INPUT_AS_HEADER_ONLY TRUE)
				if(NOT _NBL_DISABLE_CUSTOM_COMMANDS AND CMAKE_GENERATOR MATCHES "Visual Studio")
					set(ADD_INPUT_AS_HEADER_ONLY FALSE)
				endif()
				if(ADD_INPUT_AS_HEADER_ONLY)
					list(APPEND HEADER_ONLY_LIKE "${TARGET_INPUT}")
				endif()
				if(NBL_NSC_OUT_FILES AND NOT CMAKE_CONFIGURATION_TYPES)
					list(APPEND HEADER_ONLY_LIKE ${NBL_NSC_OUT_FILES})
				endif()
				if(HEADER_ONLY_LIKE AND IMPL_HLSL_GLOB)
					foreach(_HLSL_SOURCE IN LISTS IMPL_HLSL_GLOB)
						list(REMOVE_ITEM HEADER_ONLY_LIKE "${_HLSL_SOURCE}")
					endforeach()
				endif()
				if(HEADER_ONLY_LIKE)
					list(REMOVE_DUPLICATES HEADER_ONLY_LIKE)
					target_sources(${IMPL_TARGET} PRIVATE ${HEADER_ONLY_LIKE})
					set_source_files_properties(${HEADER_ONLY_LIKE} PROPERTIES 
						HEADER_FILE_ONLY ON
					)
				endif()
				set(ADD_CONFIG_AS_HEADER_ONLY TRUE)
				if(NOT _NBL_DISABLE_CUSTOM_COMMANDS)
					if(CMAKE_GENERATOR MATCHES "Visual Studio" AND NBL_NSC_MAIN_DEPENDENCY STREQUAL "${CONFIG_FILE}")
						set(ADD_CONFIG_AS_HEADER_ONLY FALSE)
					endif()
				endif()
				if(ADD_CONFIG_AS_HEADER_ONLY)
					target_sources(${IMPL_TARGET} PRIVATE "${CONFIG_FILE}")
					set_source_files_properties("${CONFIG_FILE}" PROPERTIES
						GENERATED TRUE
						HEADER_FILE_ONLY ON
					)
					if(CMAKE_GENERATOR MATCHES "Visual Studio")
						set_source_files_properties("${CONFIG_FILE}" PROPERTIES
							VS_EXCLUDED_FROM_BUILD TRUE
							VS_TOOL_OVERRIDE "None"
						)
					endif()
				endif()
				if(NOT _NBL_DISABLE_CUSTOM_COMMANDS)
					if(CMAKE_CONFIGURATION_TYPES)
						foreach(_CFG IN LISTS CMAKE_CONFIGURATION_TYPES)
							if(_CFG STREQUAL "")
								continue()
							endif()
							set(TARGET_OUTPUT_IDE "${IMPL_BINARY_DIR}/${_CFG}/${HASHED_KEY}")
							set(TARGET_OUTPUT_IDE_PREPROCESSED "${TARGET_OUTPUT_IDE}.pre.hlsl")
							if(NSC_CACHE_DIR)
								file(RELATIVE_PATH TARGET_OUTPUT_IDE_REL "${IMPL_BINARY_DIR}" "${TARGET_OUTPUT_IDE}")
								set(TARGET_OUTPUT_IDE_CACHE "${NBL_NSC_CACHE_ROOT}/${TARGET_OUTPUT_IDE_REL}.ppcache")
								set(TARGET_OUTPUT_IDE_PRECACHE "${NBL_NSC_CACHE_ROOT}/${TARGET_OUTPUT_IDE_REL}.ppcache.pre")
							else()
								set(TARGET_OUTPUT_IDE_CACHE "${TARGET_OUTPUT_IDE}.ppcache")
								set(TARGET_OUTPUT_IDE_PRECACHE "${TARGET_OUTPUT_IDE}.ppcache.pre")
							endif()
							set(NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE}" "${TARGET_OUTPUT_IDE}.log")
							if(NSC_USE_DEPFILE)
								list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE}.dep")
							endif()
							if(NSC_SHADER_CACHE)
								list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE_CACHE}")
							endif()
							set(ADD_PREPROCESSED_IDE TRUE)
							if(NSC_PREPROCESS_CACHE)
								list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE_PRECACHE}")
								if(ADD_PREPROCESSED_IDE)
									list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE_PREPROCESSED}")
								endif()
							endif()
							list(REMOVE_DUPLICATES NBL_NSC_OUT_FILES_IDE)
							target_sources(${IMPL_TARGET} PRIVATE ${NBL_NSC_OUT_FILES_IDE})
							set_source_files_properties(${NBL_NSC_OUT_FILES_IDE} PROPERTIES
								HEADER_FILE_ONLY ON
								GENERATED TRUE
							)
							if(NSC_SHADER_CACHE)
								set_source_files_properties("${TARGET_OUTPUT_IDE_CACHE}" PROPERTIES HEADER_FILE_ONLY OFF)
							endif()
							if(NSC_PREPROCESS_CACHE)
								set_source_files_properties("${TARGET_OUTPUT_IDE_PRECACHE}" PROPERTIES HEADER_FILE_ONLY OFF)
								if(ADD_PREPROCESSED_IDE)
									set_source_files_properties("${TARGET_OUTPUT_IDE_PREPROCESSED}" PROPERTIES HEADER_FILE_ONLY ON)
									if(CMAKE_GENERATOR MATCHES "Visual Studio")
										set_source_files_properties("${TARGET_OUTPUT_IDE_PREPROCESSED}" PROPERTIES
											VS_EXCLUDED_FROM_BUILD TRUE
											VS_TOOL_OVERRIDE "None"
										)
									endif()
								endif()
							endif()
							source_group("${OUT}/${_CFG}" FILES ${NBL_NSC_OUT_FILES_IDE})
						endforeach()
					else()
						set(TARGET_OUTPUT_IDE "${IMPL_BINARY_DIR}/${HASHED_KEY}")
						set(TARGET_OUTPUT_IDE_PREPROCESSED "${TARGET_OUTPUT_IDE}.pre.hlsl")
						if(NSC_CACHE_DIR)
							file(RELATIVE_PATH TARGET_OUTPUT_IDE_REL "${IMPL_BINARY_DIR}" "${TARGET_OUTPUT_IDE}")
							set(TARGET_OUTPUT_IDE_CACHE "${NBL_NSC_CACHE_ROOT}/${TARGET_OUTPUT_IDE_REL}.ppcache")
							set(TARGET_OUTPUT_IDE_PRECACHE "${NBL_NSC_CACHE_ROOT}/${TARGET_OUTPUT_IDE_REL}.ppcache.pre")
						else()
							set(TARGET_OUTPUT_IDE_CACHE "${TARGET_OUTPUT_IDE}.ppcache")
							set(TARGET_OUTPUT_IDE_PRECACHE "${TARGET_OUTPUT_IDE}.ppcache.pre")
						endif()
						set(NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE}" "${TARGET_OUTPUT_IDE}.log")
						if(NSC_USE_DEPFILE)
								list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE}.dep")
						endif()
						if(NSC_SHADER_CACHE)
								list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE_CACHE}")
						endif()
						set(ADD_PREPROCESSED_IDE TRUE)
						if(NSC_PREPROCESS_CACHE)
								list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE_PRECACHE}")
								if(ADD_PREPROCESSED_IDE)
									list(APPEND NBL_NSC_OUT_FILES_IDE "${TARGET_OUTPUT_IDE_PREPROCESSED}")
								endif()
						endif()
						list(REMOVE_DUPLICATES NBL_NSC_OUT_FILES_IDE)
						target_sources(${IMPL_TARGET} PRIVATE ${NBL_NSC_OUT_FILES_IDE})
						set_source_files_properties(${NBL_NSC_OUT_FILES_IDE} PROPERTIES
							HEADER_FILE_ONLY ON
							GENERATED TRUE
						)
						if(NSC_SHADER_CACHE)
							set_source_files_properties("${TARGET_OUTPUT_IDE_CACHE}" PROPERTIES HEADER_FILE_ONLY OFF)
						endif()
						if(NSC_PREPROCESS_CACHE)
							set_source_files_properties("${TARGET_OUTPUT_IDE_PRECACHE}" PROPERTIES HEADER_FILE_ONLY OFF)
							if(ADD_PREPROCESSED_IDE)
								set_source_files_properties("${TARGET_OUTPUT_IDE_PREPROCESSED}" PROPERTIES HEADER_FILE_ONLY ON)
								if(CMAKE_GENERATOR MATCHES "Visual Studio")
									set_source_files_properties("${TARGET_OUTPUT_IDE_PREPROCESSED}" PROPERTIES
										VS_EXCLUDED_FROM_BUILD TRUE
										VS_TOOL_OVERRIDE "None"
									)
								endif()
							endif()
						endif()
						source_group("${OUT}" FILES ${NBL_NSC_OUT_FILES_IDE})
					endif()
				endif()

				set_source_files_properties("${TARGET_OUTPUT}" PROPERTIES
					NBL_SPIRV_REGISTERED_INPUT "${TARGET_INPUT}"
					NBL_SPIRV_PERMUTATION_CONFIG "${CONFIG_FILE}"
					NBL_SPIRV_BINARY_DIR "${IMPL_BINARY_DIR}"
					NBL_SPIRV_ACCESS_KEY "${FINAL_KEY_REL_PATH}"
				)

				set_property(TARGET ${IMPL_TARGET} APPEND PROPERTY NBL_SPIRV_OUTPUTS "${TARGET_OUTPUT}")
				return()
			endif()

			list(GET CAP_NAMES ${CAP_INDEX} CURRENT_CAP)
			list(GET CAP_TYPES ${CAP_INDEX} CURRENT_TYPE)
			list(GET CAP_KINDS ${CAP_INDEX} CURRENT_KIND)
			set(VAR_NAME "CAP_VALUES_${CAP_INDEX}")
			set(VALUES "${${VAR_NAME}}")

			set(KEY_PREFIX ".")
			if(CAP_INDEX EQUAL 0)
				set(KEY_PREFIX "__${CURRENT_KIND}.")
			else()
				math(EXPR PREV_INDEX "${CAP_INDEX} - 1")
				list(GET CAP_KINDS ${PREV_INDEX} PREV_KIND)
				if(NOT CURRENT_KIND STREQUAL PREV_KIND)
					set(KEY_PREFIX "__${CURRENT_KIND}.")
				endif()
			endif()
			foreach(V IN LISTS VALUES)
				set(NEW_PREFIX "${PREFIX}${KEY_PREFIX}${CURRENT_CAP}_${V}")
				set(NEW_VALUES "${CAPS_VALUES_PART}")
				list(APPEND NEW_VALUES "${V}")
				math(EXPR NEXT_INDEX "${CAP_INDEX} + 1")
				GENERATE_KEYS("${NEW_PREFIX}" "${NEXT_INDEX}" ${NEW_VALUES})
			endforeach()
		endfunction()

       	GENERATE_KEYS("" 0)

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

	source_group("${IN}/autogen" FILES ${CONFIGS})
	source_group("${IN}" FILES ${INPUTS})
	if(IMPL_HLSL_GLOB AND INPUTS)
		set(_NBL_INPUTS_ABS "")
		foreach(_IN_FILE IN LISTS INPUTS)
			get_filename_component(_IN_ABS "${_IN_FILE}" ABSOLUTE)
			string(TOLOWER "${_IN_ABS}" _IN_ABS_LOWER)
			list(APPEND _NBL_INPUTS_ABS "${_IN_ABS_LOWER}")
		endforeach()
		set(_NBL_HLSL_FILTERED "")
		foreach(_HLSL_FILE IN LISTS IMPL_HLSL_GLOB)
			get_filename_component(_HLSL_ABS "${_HLSL_FILE}" ABSOLUTE)
			string(TOLOWER "${_HLSL_ABS}" _HLSL_ABS_LOWER)
			list(FIND _NBL_INPUTS_ABS "${_HLSL_ABS_LOWER}" _HLSL_INDEX)
			if(_HLSL_INDEX EQUAL -1)
				list(APPEND _NBL_HLSL_FILTERED "${_HLSL_FILE}")
			endif()
		endforeach()
		set(IMPL_HLSL_GLOB "${_NBL_HLSL_FILTERED}")
	endif()
	if(IMPL_HLSL_GLOB)
		target_sources(${IMPL_TARGET} PRIVATE ${IMPL_HLSL_GLOB})
		set_source_files_properties(${IMPL_HLSL_GLOB} PROPERTIES 
			HEADER_FILE_ONLY ON
		)
		if(CMAKE_GENERATOR MATCHES "Visual Studio")
			set_source_files_properties(${IMPL_HLSL_GLOB} PROPERTIES
				VS_EXCLUDED_FROM_BUILD TRUE
				VS_TOOL_OVERRIDE "None"
			)
		endif()
		source_group("HLSL Files" FILES ${IMPL_HLSL_GLOB})
	endif()

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
