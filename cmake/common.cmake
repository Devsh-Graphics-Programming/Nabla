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

function(nbl_handle_dll_definitions _TARGET_ _SCOPE_)
	if(NOT TARGET Nabla)
		message(FATAL_ERROR "Internal error, Nabla target must be defined!")
	endif()
	
	if(NOT TARGET ${_TARGET_})
		message(FATAL_ERROR "Internal error, requsted \"${_TARGET_}\" is not defined!")
	endif()

	if(NBL_DYNAMIC_MSVC_RUNTIME)
		set(_NABLA_OUTPUT_DIR_ "${NBL_ROOT_PATH_BINARY}/src/nbl/$<CONFIG>/devshgraphicsprogramming.nabla")
		
		target_compile_definitions(${_TARGET_} ${_SCOPE_} 
			_NABLA_DLL_NAME_="$<TARGET_FILE_NAME:Nabla>";_NABLA_OUTPUT_DIR_="${_NABLA_OUTPUT_DIR_}";_NABLA_INSTALL_DIR_="${CMAKE_INSTALL_PREFIX}"
		)
	endif()
	
	target_compile_definitions(${_TARGET_} ${_SCOPE_} 
		_DXC_DLL_="${DXC_DLL}"
	)
endfunction()

function(nbl_handle_runtime_lib_properties _TARGET_)
	if(NOT TARGET ${_TARGET_})
		message(FATAL_ERROR "Internal error, requsted \"${_TARGET_}\" is not defined!")
	endif()

	if(NBL_DYNAMIC_MSVC_RUNTIME)
		set_target_properties(${_TARGET_} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
	else()
		set_target_properties(${_TARGET_} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	endif()
endfunction()

# Macro creating project for an executable
# Project and target get its name from directory when this macro gets executed (truncating number in the beginning of the name and making all lower case)
# Created because of common cmake code for examples and tools
macro(nbl_create_executable_project _EXTRA_SOURCES _EXTRA_OPTIONS _EXTRA_INCLUDES _EXTRA_LIBS)
	get_filename_component(_NBL_PROJECT_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
	include("scripts/nbl/projectTargetName") # sets EXECUTABLE_NAME
	
	if(MSVC)
		set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${EXECUTABLE_NAME})
	endif()
	
	project(${EXECUTABLE_NAME})

	if(ANDROID)
		add_library(${EXECUTABLE_NAME} SHARED main.cpp ${_EXTRA_SOURCES})
	else()
		set(NBL_EXECUTABLE_SOURCES
			main.cpp
			${_EXTRA_SOURCES}
		)
		
		add_executable(${EXECUTABLE_NAME} ${NBL_EXECUTABLE_SOURCES})
		nbl_handle_runtime_lib_properties(${EXECUTABLE_NAME})
		
		if(WIN32 AND MSVC)
			if(NBL_DYNAMIC_MSVC_RUNTIME)
				target_link_options(${EXECUTABLE_NAME} PUBLIC "/DELAYLOAD:$<TARGET_FILE_NAME:Nabla>")
			endif()
			
			target_link_options(${EXECUTABLE_NAME} PUBLIC "/DELAYLOAD:dxcompiler.dll")
		endif()
	endif()
	
	nbl_handle_dll_definitions(${EXECUTABLE_NAME} PUBLIC)

	target_compile_definitions(${EXECUTABLE_NAME} PUBLIC _NBL_APP_NAME_="${EXECUTABLE_NAME}")
	
	if("${EXECUTABLE_NAME}" STREQUAL commonpch)
		add_dependencies(${EXECUTABLE_NAME} Nabla)
	else()
		string(FIND "${_NBL_PROJECT_DIRECTORY_}" "${NBL_ROOT_PATH}/examples_tests" _NBL_FOUND_)
		
		if(NOT "${_NBL_FOUND_}" STREQUAL "-1") # the call was made for a target defined in examples_tests, request common api PCH
			if(NOT TARGET ${NBL_EXECUTABLE_COMMON_API_TARGET})
				message(FATAL_ERROR "Internal error, NBL_EXECUTABLE_COMMON_API_TARGET target must be defined to create an example target!")
			endif()
		
			add_dependencies(${EXECUTABLE_NAME} ${NBL_EXECUTABLE_COMMON_API_TARGET})
			target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${NBL_EXECUTABLE_COMMON_API_TARGET})
			target_precompile_headers("${EXECUTABLE_NAME}" REUSE_FROM "${NBL_EXECUTABLE_COMMON_API_TARGET}")
		endif()
	endif()
		
	target_include_directories(${EXECUTABLE_NAME}
		PUBLIC "${NBL_ROOT_PATH}/examples_tests/common"
		PUBLIC "${NBL_ROOT_PATH_BINARY}/include"
		PUBLIC ../../include # in macro.. relative to what? TODO: correct
		PRIVATE ${_EXTRA_INCLUDES}
	)
	target_link_libraries(${EXECUTABLE_NAME} PUBLIC Nabla ${_EXTRA_LIBS})

	add_compile_options(${_EXTRA_OPTIONS})

	if(NBL_SANITIZE_ADDRESS)
		if(MSVC)
			target_compile_options(${EXECUTABLE_NAME} PUBLIC /fsanitize=address)
		else()
			target_compile_options(${EXECUTABLE_NAME} PUBLIC -fsanitize=address)
		endif()
	endif()
	
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		# add_compile_options("-msse4.2 -mfpmath=sse") ????
		add_compile_options(
			"$<$<CONFIG:DEBUG>:-fstack-protector-all>"
		)
	
		set(COMMON_LINKER_OPTIONS "-msse4.2 -mfpmath=sse -fuse-ld=gold")
		set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${COMMON_LINKER_OPTIONS}")
		set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${COMMON_LINKER_OPTIONS} -fstack-protector-strong")
		if (NBL_GCC_SANITIZE_ADDRESS)
			set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
		endif()
		if (NBL_GCC_SANITIZE_THREAD)
			set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=thread")
		endif()
		if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 6.1)
			add_compile_options(-Wno-error=ignored-attributes)
		endif()
	endif()

	nbl_adjust_flags(TARGET ${EXECUTABLE_NAME} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)	
	nbl_adjust_definitions() # macro defined in root CMakeLists
	add_definitions(-D_NBL_PCH_IGNORE_PRIVATE_HEADERS)

	set_target_properties(${EXECUTABLE_NAME} PROPERTIES DEBUG_POSTFIX _d)
	set_target_properties(${EXECUTABLE_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX _rwdi)
	set_target_properties(${EXECUTABLE_NAME}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
		RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_SOURCE_DIR}/bin"
		RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
		VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin" # for visual studio
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

	target_include_directories(${LIB_NAME}
		PUBLIC $<TARGET_PROPERTY:Nabla,INCLUDE_DIRECTORIES>
		PRIVATE ${LIB_INCLUDES}
	)
	
	if(NBL_EMBED_BUILTIN_RESOURCES)
		get_target_property(_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_ nblBuiltinResourceData BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
		
		target_include_directories(${LIB_NAME}
			PUBLIC ${_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_}
		)
	endif()
	
	add_dependencies(${LIB_NAME} Nabla)
	target_link_libraries(${LIB_NAME} PUBLIC Nabla)
	target_compile_options(${LIB_NAME} PUBLIC ${LIB_OPTIONS})
	target_compile_definitions(${LIB_NAME} PUBLIC ${DEF_OPTIONS})
	
	nbl_handle_dll_definitions(${LIB_NAME} PUBLIC)
	nbl_handle_runtime_lib_properties(${LIB_NAME})

	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options(
			"$<$<CONFIG:DEBUG>:-fstack-protector-all>"
		)

		set(COMMON_LINKER_OPTIONS "-msse4.2 -mfpmath=sse -fuse-ld=gold")
		set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${COMMON_LINKER_OPTIONS}")
		set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${COMMON_LINKER_OPTIONS} -fstack-protector-strong -fsanitize=address")
	endif()

	nbl_adjust_flags(TARGET ${LIB_NAME} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
	nbl_adjust_definitions() # macro defined in root CMakeLists

	set_target_properties(${LIB_NAME} PROPERTIES DEBUG_POSTFIX "")
	set_target_properties(${LIB_NAME} PROPERTIES RELWITHDEBINFO_POSTFIX _rwdb)
	set_target_properties(${LIB_NAME}
		PROPERTIES
		RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
	)
	if(MSVC)
		set_target_properties(${LIB_NAME}
			PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY_DEBUG "${PROJECT_SOURCE_DIR}/bin"
			RUNTIME_OUTPUT_DIRECTORY_RELEASE "${PROJECT_SOURCE_DIR}/bin"
			RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${PROJECT_SOURCE_DIR}/bin"
			VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/bin" # seems like has no effect
		)
	endif()
	
	nbl_install_file_spec(${LIB_HEADERS} "nbl/ext/${EXT_NAME}")	
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

function(nbl_get_conf_dir _OUTVAR _CONFIG)
	string(TOLOWER ${_CONFIG} CONFIG)
	set(${_OUTVAR} "${NBL_ROOT_PATH_BINARY}/include/nbl/config/${CONFIG}" PARENT_SCOPE)
endfunction()

macro(nbl_generate_conf_files)
	nbl_get_conf_dir(NBL_CONF_DIR_DEBUG Debug)
	nbl_get_conf_dir(NBL_CONF_DIR_RELEASE Release)
	nbl_get_conf_dir(NBL_CONF_DIR_RELWITHDEBINFO RelWithDebInfo)

	set(_NBL_DEBUG 0)
	set(_NBL_RELWITHDEBINFO 0)

	configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${NBL_CONF_DIR_RELEASE}/BuildConfigOptions.h.conf")
	file(GENERATE OUTPUT "${NBL_CONF_DIR_RELEASE}/BuildConfigOptions.h" INPUT "${NBL_CONF_DIR_RELEASE}/BuildConfigOptions.h.conf" CONDITION $<CONFIG:Release>)

	set(_NBL_DEBUG 0)
	set(_NBL_RELWITHDEBINFO 1)
	
	configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${NBL_CONF_DIR_RELWITHDEBINFO}/BuildConfigOptions.h.conf")
	file(GENERATE OUTPUT "${NBL_CONF_DIR_RELWITHDEBINFO}/BuildConfigOptions.h" INPUT "${NBL_CONF_DIR_RELWITHDEBINFO}/BuildConfigOptions.h.conf" CONDITION $<CONFIG:RelWithDebInfo>)

	set(_NBL_DEBUG 1)
	set(_NBL_RELWITHDEBINFO 0)

	configure_file("${NBL_ROOT_PATH}/include/nbl/config/BuildConfigOptions.h.in" "${NBL_CONF_DIR_DEBUG}/BuildConfigOptions.h.conf")
	file(GENERATE OUTPUT "${NBL_CONF_DIR_DEBUG}/BuildConfigOptions.h" INPUT "${NBL_CONF_DIR_DEBUG}/BuildConfigOptions.h.conf" CONDITION $<CONFIG:Debug>)

	unset(NBL_CONF_DIR_DEBUG)
	unset(NBL_CONF_DIR_RELEASE)
	unset(NBL_CONF_DIR_RELWITHDEBINFO)
endmacro()

###########################################
# Nabla install rules, directory structure:
#
# -	$<CONFIG>/include 		(header files)
# - $<CONFIG>/lib 			(import/static/shared libraries)
# - $<CONFIG>/runtime 		(DLLs/PDBs)
# - $<CONFIG>/exe			(executables and media)
#
# If $<CONFIG> == Release, then the directory structure doesn't begin with $<CONFIG>

set(_NBL_CPACK_PACKAGE_RELATIVE_ENTRY_ "./$<$<NOT:$<STREQUAL:$<CONFIG>,Release>>:$<LOWER_CASE:$<CONFIG>>>")

function(nbl_install_headers_spec _HEADERS _BASE_HEADERS_DIR)
	foreach (file ${_HEADERS})
		file(RELATIVE_PATH dir ${_BASE_HEADERS_DIR} ${file})
		get_filename_component(dir ${dir} DIRECTORY)
		install(FILES ${file} DESTINATION include/${dir} CONFIGURATIONS Release COMPONENT Headers)
		install(FILES ${file} DESTINATION debug/include/${dir} CONFIGURATIONS Debug COMPONENT Headers)
		install(FILES ${file} DESTINATION relwithdebinfo/include/${dir} CONFIGURATIONS RelWithDebInfo COMPONENT Headers)
	endforeach()
endfunction()

function(nbl_install_headers _HEADERS)
	if(NOT DEFINED NBL_ROOT_PATH)
		message(FATAL_ERROR "NBL_ROOT_PATH isn't defined!")
	endif()

	nbl_install_headers_spec("${_HEADERS}" "${NBL_ROOT_PATH}/include")
endfunction()

function(nbl_install_file_spec _FILES _RELATIVE_DESTINATION)
	install(FILES ${_FILES} DESTINATION include/${_RELATIVE_DESTINATION} CONFIGURATIONS Release COMPONENT Headers)
	install(FILES ${_FILES} DESTINATION debug/include/${_RELATIVE_DESTINATION} CONFIGURATIONS Debug COMPONENT Headers)
	install(FILES ${_FILES} DESTINATION relwithdebinfo/include/${_RELATIVE_DESTINATION} CONFIGURATIONS RelWithDebInfo COMPONENT Headers)
endfunction()

function(nbl_install_file _FILES)
	nbl_install_file_spec("${_FILES}" "")
endfunction()

function(nbl_install_dir_spec _DIR _RELATIVE_DESTINATION)
	install(DIRECTORY ${_DIR} DESTINATION include/${_RELATIVE_DESTINATION} CONFIGURATIONS Release COMPONENT Headers)
	install(DIRECTORY ${_DIR} DESTINATION debug/include/${_RELATIVE_DESTINATION} CONFIGURATIONS Debug COMPONENT Headers)
	install(DIRECTORY ${_DIR} DESTINATION relwithdebinfo/include/${_RELATIVE_DESTINATION} CONFIGURATIONS RelWithDebInfo COMPONENT Headers)
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

function(nbl_install_config_header _CONF_HDR_NAME)
	nbl_get_conf_dir(dir_deb Debug)
	nbl_get_conf_dir(dir_rel Release)
	nbl_get_conf_dir(dir_relWithDebInfo RelWithDebInfo)
	set(file_deb "${dir_deb}/${_CONF_HDR_NAME}")
	set(file_rel "${dir_rel}/${_CONF_HDR_NAME}")
	set(file_relWithDebInfo "${dir_relWithDebInfo}/${_CONF_HDR_NAME}")
	install(FILES ${file_rel} DESTINATION include CONFIGURATIONS Release)
	install(FILES ${file_deb} DESTINATION debug/include CONFIGURATIONS Debug)
	install(FILES ${file_relWithDebInfo} DESTINATION relwithdebinfo/include CONFIGURATIONS RelWithDebInfo)
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

macro(glue_source_definitions NBL_TARGET NBL_REFERENCE_RETURN_VARIABLE)
	macro(NBL_INSERT_DEFINITIONS _NBL_DEFINITIONS_)
		string(FIND "${_NBL_DEFINITIONS_}" "NOTFOUND" CHECK)
			if(${CHECK} EQUAL -1)
				list(APPEND TESTEST ${_NBL_DEFINITIONS_})
			endif()
	endmacro()
		
	get_directory_property(NBL_DIRECTORY_DEFINITIONS COMPILE_DEFINITIONS)

	if(DEFINED NBL_DIRECTORY_DEFINITIONS)
		NBL_INSERT_DEFINITIONS("${NBL_DIRECTORY_DEFINITIONS}")
	endif()
	
	get_target_property(NBL_COMPILE_DEFS ${NBL_TARGET} COMPILE_DEFINITIONS)
	if(DEFINED NBL_COMPILE_DEFS)
		foreach(def IN LISTS NBL_COMPILE_DEFS)
			NBL_INSERT_DEFINITIONS(${def})
		endforeach()
	endif()
	
	foreach(trgt IN LISTS _NBL_3RDPARTY_TARGETS_)			 
			 get_target_property(NBL_COMPILE_DEFS ${trgt} COMPILE_DEFINITIONS)
			 
			 if(DEFINED NBL_COMPILE_DEFS)
				NBL_INSERT_DEFINITIONS(${NBL_COMPILE_DEFS})
			 endif()
	endforeach()
	
	foreach(def IN LISTS TESTEST)	
		string(FIND "${def}" "-D" CHECK)
			if(${CHECK} EQUAL -1)
				list(APPEND ${NBL_REFERENCE_RETURN_VARIABLE} ${def})
			else()
				string(LENGTH "-D" _NBL_D_LENGTH_)
				string(LENGTH ${def} _NBL_DEFINITION_LENGTH_)
				math(EXPR _NBL_DEFINITION_WITHOUT_D_LENGTH_ "${_NBL_DEFINITION_LENGTH_} - ${_NBL_D_LENGTH_}" OUTPUT_FORMAT DECIMAL)
				string(SUBSTRING ${def} ${_NBL_D_LENGTH_} ${_NBL_DEFINITION_WITHOUT_D_LENGTH_} _NBL_DEFINITION_WITHOUT_D_)
				
				list(APPEND ${NBL_REFERENCE_RETURN_VARIABLE} ${_NBL_DEFINITION_WITHOUT_D_})
			endif()
	endforeach()
	
	list(REMOVE_DUPLICATES ${NBL_REFERENCE_RETURN_VARIABLE})
	
	foreach(_NBL_DEF_ IN LISTS ${NBL_REFERENCE_RETURN_VARIABLE})
		string(FIND "${_NBL_DEF_}" "=" _NBL_POSITION_ REVERSE)
		
		# put target compile definitions without any value into wrapper file
		if(_NBL_POSITION_ STREQUAL -1)
			if(NOT ${_NBL_DEF_} STREQUAL "__NBL_BUILDING_NABLA__")
				string(APPEND WRAPPER_CODE 
					"#ifndef ${_NBL_DEF_}\n"
					"#define ${_NBL_DEF_}\n"
					"#endif // ${_NBL_DEF_}\n\n"
				)
			endif()
		else()
			# put target compile definitions with an assigned value into wrapper file
			string(SUBSTRING "${_NBL_DEF_}" 0 ${_NBL_POSITION_} _NBL_CLEANED_DEF_)
			
			string(LENGTH "${_NBL_DEF_}" _NBL_DEF_LENGTH_)
			math(EXPR _NBL_SHIFTED_POSITION_ "${_NBL_POSITION_} + 1" OUTPUT_FORMAT DECIMAL)
			math(EXPR _NBL_DEF_VALUE_LENGTH_ "${_NBL_DEF_LENGTH_} - ${_NBL_SHIFTED_POSITION_}" OUTPUT_FORMAT DECIMAL)
			string(SUBSTRING "${_NBL_DEF_}" ${_NBL_SHIFTED_POSITION_} ${_NBL_DEF_VALUE_LENGTH_} _NBL_DEF_VALUE_)
			
			string(APPEND WRAPPER_CODE 
				"#ifndef ${_NBL_CLEANED_DEF_}\n"
				"#define ${_NBL_CLEANED_DEF_} ${_NBL_DEF_VALUE_}\n"
				"#endif // ${_NBL_CLEANED_DEF_}\n\n"
			)
		endif()
	endforeach()
	
	set(${NBL_REFERENCE_RETURN_VARIABLE} "${WRAPPER_CODE}")
endmacro()

macro(write_source_definitions NBL_FILE NBL_WRAPPER_CODE_TO_WRITE)
	file(WRITE "${NBL_FILE}" "${NBL_WRAPPER_CODE_TO_WRITE}")
endmacro()