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

include(ProcessorCount)
set(_NBL_CPACK_PACKAGE_RELATIVE_ENTRY_ "$<$<NOT:$<STREQUAL:$<CONFIG>,Release>>:$<LOWER_CASE:$<CONFIG>>>" CACHE INTERNAL "")

# submodule managment
function(update_git_submodule _PATH)
	execute_process(COMMAND git submodule update --init --recursive ${_PATH}
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
	)
endfunction()

# TODO: REDO THIS WHOLE THING AS FUNCTIONS
# https://github.com/buildaworldnet/IrrlichtBAW/issues/311 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

# Macro creating project for an executable
# Project and target get its name from directory when this macro gets executed (truncating number in the beginning of the name and making all lower case)
# Created because of common cmake code for examples and tools
macro(nbl_create_executable_project _EXTRA_SOURCES _EXTRA_OPTIONS _EXTRA_INCLUDES _EXTRA_LIBS _PCH_TARGET) # TODO remove _PCH_TARGET
	set(_NBL_PROJECT_DIRECTORY_ "${CMAKE_CURRENT_SOURCE_DIR}")
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
		
		if(NBL_DYNAMIC_MSVC_RUNTIME)
			set_property(TARGET ${EXECUTABLE_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
			
			if(WIN32 AND MSVC)				
				target_link_options(${EXECUTABLE_NAME} PUBLIC "/DELAYLOAD:$<TARGET_FILE_NAME:Nabla>")
			endif()
		else()
			set_property(TARGET ${EXECUTABLE_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
		endif()
		
		if(WIN32 AND MSVC)
			target_link_options(${EXECUTABLE_NAME} PUBLIC "/DELAYLOAD:dxcompiler.dll")
		endif()
	endif()
	
	if("${EXECUTABLE_NAME}" STREQUAL commonpch)
		add_dependencies(${EXECUTABLE_NAME} Nabla)
	else()
		if(NOT TARGET ${NBL_EXECUTABLE_COMMON_API_TARGET})
			message(FATAL_ERROR "Internal error, NBL_EXECUTABLE_COMMON_API_TARGET target must be defined!")
		endif()
	
		add_dependencies(${EXECUTABLE_NAME} ${NBL_EXECUTABLE_COMMON_API_TARGET})
		target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${NBL_EXECUTABLE_COMMON_API_TARGET})
		target_precompile_headers("${EXECUTABLE_NAME}" REUSE_FROM "${NBL_EXECUTABLE_COMMON_API_TARGET}")
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
		target_compile_options(${EXECUTABLE_NAME} PUBLIC "-fsanitize=address /fsanitize=address")
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

	# https://github.com/buildaworldnet/IrrlichtBAW/issues/298 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
	nbl_adjust_flags() # macro defined in root CMakeLists
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
endmacro()

macro(nbl_create_ext_library_project EXT_NAME LIB_HEADERS LIB_SOURCES LIB_INCLUDES LIB_OPTIONS DEF_OPTIONS)
	set(LIB_NAME "NblExt${EXT_NAME}")
	project(${LIB_NAME})

	add_library(${LIB_NAME} ${LIB_SOURCES})
	get_target_property(_NBL_NABLA_TARGET_BINARY_DIR_ Nabla BINARY_DIR)

	# TODO: correct those bugs, use generator expressions
	target_include_directories(${LIB_NAME}
		PUBLIC ${_NBL_NABLA_TARGET_BINARY_DIR_}/build/import
		PUBLIC ${CMAKE_BINARY_DIR}/include/nbl/config/debug
		PUBLIC ${CMAKE_BINARY_DIR}/include/nbl/config/release
		PUBLIC ${CMAKE_BINARY_DIR}/include/nbl/config/relwithdebinfo
		PUBLIC ${CMAKE_SOURCE_DIR}/include
		PUBLIC ${CMAKE_SOURCE_DIR}/src
		PUBLIC ${CMAKE_SOURCE_DIR}/source/Nabla
		PRIVATE ${LIB_INCLUDES}
	)
	
	if(NBL_EMBED_BUILTIN_RESOURCES)
		get_target_property(_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_ nblBuiltinResourceData BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
		
		target_include_directories(${LIB_NAME}
			PUBLIC ${_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_}
		)
	endif()
	
	if(NBL_DYNAMIC_MSVC_RUNTIME)
		if(WIN32 AND MSVC)
			set(_NABLA_OUTPUT_DIR_ "${NBL_ROOT_PATH_BINARY}/src/nbl/$<CONFIG>/devshgraphicsprogramming.nabla")
			
			target_compile_definitions(${LIB_NAME} PUBLIC 
				_NABLA_DLL_NAME_="$<TARGET_FILE_NAME:Nabla>";_NABLA_OUTPUT_DIR_="${_NABLA_OUTPUT_DIR_}";_NABLA_INSTALL_DIR_="${CMAKE_INSTALL_PREFIX}"
			)
		endif()
	endif()

	if(WIN32 AND MSVC)
		target_compile_definitions(${LIB_NAME} PUBLIC 
			_DXC_DLL_="${DXC_DLL}"
		)
	endif()
	
	add_dependencies(${LIB_NAME} Nabla)
	target_link_libraries(${LIB_NAME} PUBLIC Nabla)
	target_compile_options(${LIB_NAME} PUBLIC ${LIB_OPTIONS})
	target_compile_definitions(${LIB_NAME} PUBLIC ${DEF_OPTIONS})
	if(NBL_DYNAMIC_MSVC_RUNTIME)
		set_target_properties(${LIB_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
	else()
		set_target_properties(${LIB_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	endif()

	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		add_compile_options(
			"$<$<CONFIG:DEBUG>:-fstack-protector-all>"
		)

		set(COMMON_LINKER_OPTIONS "-msse4.2 -mfpmath=sse -fuse-ld=gold")
		set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${COMMON_LINKER_OPTIONS}")
		set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${COMMON_LINKER_OPTIONS} -fstack-protector-strong -fsanitize=address")
	endif()

	# https://github.com/buildaworldnet/IrrlichtBAW/issues/298 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
	nbl_adjust_flags() # macro defined in root CMakeLists
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

function(nbl_install_media_spec _DIR _RELATIVE_DESTINATION)
	install(DIRECTORY ${_DIR} DESTINATION exe/${_RELATIVE_DESTINATION} CONFIGURATIONS Release COMPONENT Media PATTERN "Ditt-Reference-Scenes/*" EXCLUDE)
	install(DIRECTORY ${_DIR} DESTINATION debug/exe/${_RELATIVE_DESTINATION} CONFIGURATIONS Debug COMPONENT Media PATTERN "Ditt-Reference-Scenes/*" EXCLUDE)
	install(DIRECTORY ${_DIR} DESTINATION relwithdebinfo/exe/${_RELATIVE_DESTINATION} CONFIGURATIONS RelWithDebInfo COMPONENT Media PATTERN "Ditt-Reference-Scenes/*" EXCLUDE)
endfunction()

function(nbl_install_media _DIR)
	nbl_install_media_spec("${_DIR}" "")
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

# links builtin resource target to a target
# @_TARGET_@ is target name builtin resource target will be linked to
# @_BS_TARGET_@ is a builtin resource target

function(LINK_BUILTIN_RESOURCES_TO_TARGET _TARGET_ _BS_TARGET_)
	add_dependencies(${EXECUTABLE_NAME} ${_BS_TARGET_})
	target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${_BS_TARGET_})
	
	get_target_property(_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_ ${_BS_TARGET_} BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
	target_include_directories(${EXECUTABLE_NAME} PUBLIC "${_BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY_}")
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

function(NBL_UPDATE_SUBMODULES)
	macro(NBL_WRAPPER_COMMAND GIT_RELATIVE_ENTRY GIT_SUBMODULE_PATH SHOULD_RECURSIVE)
		set(SHOULD_RECURSIVE ${SHOULD_RECURSIVE})
	
		if(SHOULD_RECURSIVE)
			string(APPEND _NBL_UPDATE_SUBMODULES_COMMANDS_ "\"${GIT_EXECUTABLE}\" -C \"${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}\" submodule update --init --recursive ${GIT_SUBMODULE_PATH}\n")
		else()
			string(APPEND _NBL_UPDATE_SUBMODULES_COMMANDS_ "\"${GIT_EXECUTABLE}\" -C \"${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}\" submodule update --init ${GIT_SUBMODULE_PATH}\n")
		endif()
	endmacro()
	
	if(NBL_UPDATE_GIT_SUBMODULE)
		execute_process(COMMAND ${CMAKE_COMMAND} -E echo "All submodules are about to get updated and initialized in repository because NBL_UPDATE_GIT_SUBMODULE is turned ON!")
		set(_NBL_UPDATE_SUBMODULES_CMD_NAME_ "nbl-update-submodules")
		set(_NBL_UPDATE_SUBMODULES_CMD_FILE_ "${NBL_ROOT_PATH_BINARY}/${_NBL_UPDATE_SUBMODULES_CMD_NAME_}.cmd")
		
		if(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE)
			NBL_WRAPPER_COMMAND("" "" TRUE)
		else()
			NBL_WRAPPER_COMMAND("" ./3rdparty TRUE)
			#NBL_WRAPPER_COMMAND("" ./ci TRUE) TODO: enable it once we merge Ditt, etc
			NBL_WRAPPER_COMMAND("" ./examples_tests FALSE)
			NBL_WRAPPER_COMMAND(examples_tests ./media FALSE)
		endif()
				
		file(WRITE "${_NBL_UPDATE_SUBMODULES_CMD_FILE_}" "${_NBL_UPDATE_SUBMODULES_COMMANDS_}")

		if(WIN32)
			find_package(GitBash REQUIRED)
		
			execute_process(COMMAND "${GIT_BASH_EXECUTABLE}" "-c"
[=[
>&2 echo ""
clear
./nbl-update-submodules.cmd 2>&1 | tee nbl-update-submodules.log
sleep 1
clear
tput setaf 2; echo -e "Submodules have been updated! 
Created nbl-update-submodules.log in your build directory. 
This window will be closed in 5 seconds..."
sleep 5
]=]
				WORKING_DIRECTORY ${NBL_ROOT_PATH_BINARY}
				OUTPUT_VARIABLE _NBL_TMP_OUTPUT_
				RESULT_VARIABLE _NBL_TMP_RET_CODE_
				OUTPUT_STRIP_TRAILING_WHITESPACE
				ERROR_STRIP_TRAILING_WHITESPACE
			)

			unset(_NBL_TMP_OUTPUT_)
			unset(_NBL_TMP_RET_CODE_)
		else()
			execute_process(COMMAND "${_NBL_UPDATE_SUBMODULES_CMD_FILE_}")
		endif()
	else()
		execute_process(COMMAND ${CMAKE_COMMAND} -E echo "NBL_UPDATE_GIT_SUBMODULE is turned OFF therefore submodules won't get updated.")
	endif()
endfunction()