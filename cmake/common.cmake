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
macro(nbl_create_executable_project _EXTRA_SOURCES _EXTRA_OPTIONS _EXTRA_INCLUDES _EXTRA_LIBS)
	get_filename_component(EXECUTABLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
	string(REGEX REPLACE "^[0-9]+\." "" EXECUTABLE_NAME ${EXECUTABLE_NAME})
	string(TOLOWER ${EXECUTABLE_NAME} EXECUTABLE_NAME)
	string(MAKE_C_IDENTIFIER ${EXECUTABLE_NAME} EXECUTABLE_NAME)

	project(${EXECUTABLE_NAME})

	if(ANDROID)
		add_library(${EXECUTABLE_NAME} SHARED main.cpp ${_EXTRA_SOURCES})
	else()
		add_executable(${EXECUTABLE_NAME} main.cpp ${_EXTRA_SOURCES})
	endif()
	
	set_property(TARGET ${EXECUTABLE_NAME} PROPERTY
             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	
	# EXTRA_SOURCES is var containing non-common names of sources (if any such sources, then EXTRA_SOURCES must be set before including this cmake code)
	add_dependencies(${EXECUTABLE_NAME} Nabla)

	target_include_directories(${EXECUTABLE_NAME}
		PUBLIC ../../include
		PRIVATE ${_EXTRA_INCLUDES}
	)
	target_link_libraries(${EXECUTABLE_NAME} Nabla ${_EXTRA_LIBS}) # see, this is how you should code to resolve github issue 311

	add_compile_options(${_EXTRA_OPTIONS})
	
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
endmacro()

macro(nbl_create_ext_library_project EXT_NAME LIB_HEADERS LIB_SOURCES LIB_INCLUDES LIB_OPTIONS DEF_OPTIONS)
	set(LIB_NAME "NblExt${EXT_NAME}")
	project(${LIB_NAME})

	add_library(${LIB_NAME} ${LIB_SOURCES})
	# EXTRA_SOURCES is var containing non-common names of sources (if any such sources, then EXTRA_SOURCES must be set before including this cmake code)
	add_dependencies(${LIB_NAME} Nabla)
	
	get_target_property(_NBL_NABLA_TARGET_BINARY_DIR_ Nabla BINARY_DIR)

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
	add_dependencies(${LIB_NAME} Nabla)
	target_link_libraries(${LIB_NAME} PUBLIC Nabla)
	target_compile_options(${LIB_NAME} PUBLIC ${LIB_OPTIONS})
	target_compile_definitions(${LIB_NAME} PUBLIC ${DEF_OPTIONS})
	set_target_properties(${LIB_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

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

	install(
		FILES ${LIB_HEADERS}
		DESTINATION ./include/nbl/ext/${EXT_NAME}
		CONFIGURATIONS Release
	)
	install(
		FILES ${LIB_HEADERS}
		DESTINATION ./debug/include/nbl/ext/${EXT_NAME}
		CONFIGURATIONS Debug
	)
	install(
		FILES ${LIB_HEADERS}
		DESTINATION ./relwithdebinfo/include/nbl/ext/${EXT_NAME}
		CONFIGURATIONS RelWithDebInfo
	)
	install(
		TARGETS ${LIB_NAME}
		DESTINATION ./lib/nbl/ext/${EXT_NAME}
		CONFIGURATIONS Release
	)
	install(
		TARGETS ${LIB_NAME}
		DESTINATION ./debug/lib/nbl/ext/${EXT_NAME}
		CONFIGURATIONS Debug
	)
	install(
		TARGETS ${LIB_NAME}
		DESTINATION ./relwithdebinfo/lib/nbl/ext/${EXT_NAME}
		CONFIGURATIONS RelWithDebInfo
	)

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

# End of TODO, rest are all functions

function(nbl_get_conf_dir _OUTVAR _CONFIG)
	string(TOLOWER ${_CONFIG} CONFIG)
	set(${_OUTVAR} "${CMAKE_BINARY_DIR}/include/nbl/config/${CONFIG}" PARENT_SCOPE) # WTF TODO: change CMAKE_BINARY_DIR in future! 
endfunction()


# function for installing header files preserving directory structure
# _DEST_DIR is directory relative to CMAKE_INSTALL_PREFIX
function(nbl_install_headers _HEADERS _BASE_HEADERS_DIR)
	foreach (file ${_HEADERS})
		file(RELATIVE_PATH dir ${_BASE_HEADERS_DIR} ${file})
		get_filename_component(dir ${dir} DIRECTORY)
		install(FILES ${file} DESTINATION include/${dir} CONFIGURATIONS Release)
		install(FILES ${file} DESTINATION debug/include/${dir} CONFIGURATIONS Debug)
		install(FILES ${file} DESTINATION relwithdebinfo/include/${dir} CONFIGURATIONS RelWithDebInfo)
	endforeach()
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
	set(D8_SCRIPT "${ANDROID_BUILD_TOOLS}/d8.bat")
    #if(NOT EXISTS ${D8_SCRIPT})
        set(DEX_COMMAND ${ANDROID_BUILD_TOOLS}/dx --dex --output=bin/classes.dex ./obj)
    #else()
    #    set(DEX_COMMAND ${D8_SCRIPT} --output ./bin/ ./obj/eu/devsh/${TARGET_NAME}/*.class)
    #endif()
	#message(FATAL_ERROR "ANDROID_BUILD_TOOLS: ${ANDROID_BUILD_TOOLS}")
	#message(FATAL_ERROR "ANDROID_ANDROID_JAR_LOCATION: ${ANDROID_ANDROID_JAR_LOCATION}")
	#set(ANDROID_JAVA_RT_JAR "C:/Program Files (x86)/Java/jre1.8.0_301/lib/rt.jar")
	#message(FATAL_ERROR "ANDROID_JAR: ${ANDROID_JAR}")
	
	if(EXISTS ${ASSET_SOURCE_DIR})
		add_custom_command(
		OUTPUT ${APK_FILE}
		DEPENDS ${_TARGET}
		DEPENDS ${NBL_ANDROID_MANIFEST_FILE}
		DEPENDS ${NBL_ANDROID_LOADER_JAVA}
		DEPENDS ${KEYSTORE_FILE}
		DEPENDS ${NBL_ROOT_PATH}/android/Loader.java
		WORKING_DIRECTORY ${NBL_GEN_DIRECTORY}/$<CONFIG>
		COMMENT "Creating ${APK_FILE_NAME} ..."
		COMMAND ${CMAKE_COMMAND} -E make_directory libs/lib/x86_64
		COMMAND ${CMAKE_COMMAND} -E make_directory obj
		COMMAND ${CMAKE_COMMAND} -E make_directory bin
		COMMAND ${CMAKE_COMMAND} -E make_directory assets
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_TARGET}> libs/lib/x86_64/$<TARGET_FILE_NAME:${_TARGET}>
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${ASSET_SOURCE_DIR} assets
		COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -m -J src -M AndroidManifest.xml -I ${ANDROID_JAR}
		COMMAND ${ANDROID_JAVA_BIN}/javac -d ./obj -source 1.7 -target 1.7 -bootclasspath ${ANDROID_JAVA_RT_JAR} -classpath "${ANDROID_JAR}" -sourcepath src ${NBL_ANDROID_LOADER_JAVA}
		COMMAND ${DEX_COMMAND}
		COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -M AndroidManifest.xml -A assets -I ${ANDROID_JAR} -F ${TARGET_NAME}-unaligned.apk bin libs
		COMMAND ${ANDROID_BUILD_TOOLS}/zipalign -f 4 ${TARGET_NAME}-unaligned.apk ${APK_FILE_NAME}
		COMMAND ${ANDROID_BUILD_TOOLS}/apksigner sign --ks ${KEYSTORE_FILE} --ks-pass pass:android --key-pass pass:android --ks-key-alias ${KEY_ENTRY_ALIAS} ${APK_FILE_NAME}
		COMMAND ${CMAKE_COMMAND} -E copy ${APK_FILE_NAME} ${APK_FILE}
		COMMAND ${CMAKE_COMMAND} -E rm -rf assets
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
		WORKING_DIRECTORY ${NBL_GEN_DIRECTORY}/$<CONFIG>
		COMMENT "Creating ${APK_FILE_NAME} ..."
		COMMAND ${CMAKE_COMMAND} -E make_directory libs/lib/x86_64
		COMMAND ${CMAKE_COMMAND} -E make_directory obj
		COMMAND ${CMAKE_COMMAND} -E make_directory bin
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${_TARGET}> libs/lib/x86_64/$<TARGET_FILE_NAME:${_TARGET}>
		COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -m -J src -M AndroidManifest.xml -I ${ANDROID_JAR}
		COMMAND ${ANDROID_JAVA_BIN}/javac -d ./obj -source 1.7 -target 1.7 -bootclasspath ${ANDROID_JAVA_RT_JAR} -classpath "${ANDROID_JAR}" -sourcepath src ${NBL_ANDROID_LOADER_JAVA}
		COMMAND ${DEX_COMMAND}
		COMMAND ${ANDROID_BUILD_TOOLS}/dx --dex --output=bin/classes.dex ./obj
		COMMAND ${ANDROID_BUILD_TOOLS}/aapt package -f -M AndroidManifest.xml -I ${ANDROID_JAR} -F ${TARGET_NAME}-unaligned.apk libs
		COMMAND ${ANDROID_BUILD_TOOLS}/zipalign -f 4 ${TARGET_NAME}-unaligned.apk ${APK_FILE_NAME}
		COMMAND ${ANDROID_BUILD_TOOLS}/apksigner sign --ks ${KEYSTORE_FILE} --ks-pass pass:android --key-pass pass:android --ks-key-alias ${KEY_ENTRY_ALIAS} ${APK_FILE_NAME}
		COMMAND ${CMAKE_COMMAND} -E copy ${APK_FILE_NAME} ${APK_FILE}
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
		
		if(_NBL_POSITION_ STREQUAL -1)
			string(APPEND WRAPPER_CODE 
				"#ifndef ${_NBL_DEF_}\n"
				"#define ${_NBL_DEF_}\n"
				"#endif // ${_NBL_DEF_}\n\n"
			)
		else()
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