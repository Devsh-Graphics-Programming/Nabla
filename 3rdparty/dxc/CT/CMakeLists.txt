set(_NBL_DXC_CT_INCLUDE_ROOT_ "${NBL_ROOT_PATH}/include") # Nabla include root path
set(_NBL_DXC_CT_BINARY_BIN_ "${CMAKE_CURRENT_BINARY_DIR}") # compile-test directory

if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
	message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} _NBL_DXC_CT_INCLUDE_ROOT_ = '${_NBL_DXC_CT_INCLUDE_ROOT_}'")
	message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} _NBL_DXC_CT_BINARY_BIN_ = '${_NBL_DXC_CT_BINARY_BIN_}'")
endif()

####################################
# NBL_DXC_CT_ADD_INPUT_FILE function
# @_NBL_INPUT_REL_FILEPATH_@ is path relative to _NBL_DXC_CT_INCLUDE_ROOT_, the file must be nested in _NBL_DXC_CT_INCLUDE_ROOT_ as well to follow Nabla include rules
# @_NBL_PROFILE_@ is DXC target profile, e.g "lib_6_7"
# @_NBL_ENTRY_POINT_@ is entry point name for both HLSL and SPIRV if enabled. Leave it as empty "" string to not specify the entry point explicitly
# @_NBL_GENERATE_SPIRV_@ is a bool option for enabling SPIRV code generation 
#############################################################################

function(NBL_DXC_CT_ADD_INPUT_FILE _NBL_INPUT_REL_FILEPATH_ _NBL_PROFILE_ _NBL_ENTRY_POINT_ _NBL_GENERATE_SPIRV_)
	set(_SHADER_FILE_PATH_ "${_NBL_DXC_CT_INCLUDE_ROOT_}/${_NBL_INPUT_REL_FILEPATH_}")
	
	if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
		message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} adding _SHADER_FILE_PATH_ = '${_SHADER_FILE_PATH_}' as INPUT of DXC CT target")
	endif()
	
	cmake_path(RELATIVE_PATH _SHADER_FILE_PATH_ BASE_DIRECTORY "${_NBL_DXC_CT_INCLUDE_ROOT_}" OUTPUT_VARIABLE _SHADER_NEW_RELATIVE_FILE_PATH_)
	cmake_path(GET _SHADER_NEW_RELATIVE_FILE_PATH_ STEM LAST_ONLY _SHADER_NEW_STEAM_)
	cmake_path(REMOVE_EXTENSION _SHADER_NEW_RELATIVE_FILE_PATH_ LAST_ONLY OUTPUT_VARIABLE _SHADER_NEW_RELATIVE_OUTPUT_DIR_)
	
	set(_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_ "${_NBL_DXC_CT_BINARY_BIN_}/${_SHADER_NEW_RELATIVE_OUTPUT_DIR_}/bin")
	set(_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/${_SHADER_NEW_STEAM_}.bin")
	
	string(FIND "${_NBL_PROFILE_}" "lib" _TMP_FOUND_LIB_PROFILE_)
	if("${_TMP_FOUND_LIB_PROFILE_}" STREQUAL "-1")
		if(_NBL_GENERATE_SPIRV_)
			cmake_path(REPLACE_EXTENSION _NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_ "spv")
		endif()
	else()
		cmake_path(REPLACE_EXTENSION _NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_ "dxil")
	endif()
	unset(_TMP_FOUND_LIB_PROFILE_)
	
	# MSVC and other generators have "SetEnv" task - we need to make sure our enviroment is clean while invoking DXC compilation
	string(APPEND _TMP_CLEAN_ENV_ "set PATH=$ENV{PATH}\nset CAExcludePath=\nset LIB=\nset LIBPATH=\nset INCLUDE=\nset EXTERNAL_INCLUDE=\n")
	
	if(NOT "${_NBL_ENTRY_POINT_}" STREQUAL "")
		string(APPEND _TMP_ENTRY_POINT_ARG_ "-E ${_NBL_ENTRY_POINT_}")
	endif()

	if(_NBL_GENERATE_SPIRV_)
		string(APPEND _TMP_ENABLE_SPIRV_GEN_ARG_ "-spirv")
	endif()
	
	# include-dependency input file chain
	set(_NBL_DXC_CT_CURRENT_DEPENDS_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/depends.txt")
	set(_NBL_DXC_CT_CURRENT_IDC_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/depends.cmd")
	string(APPEND _NBL_DXC_CT_WRAPPER_IDC_COMMAND_ "${_TMP_CLEAN_ENV_}\"${DXC_EXT_EXECUTABLE}\" -HV 2021 -T ${_NBL_PROFILE_} ${_TMP_ENTRY_POINT_ARG_} ${_TMP_ENABLE_SPIRV_GEN_ARG_} -I\"${_NBL_DXC_CT_INCLUDE_ROOT_}\" -Zi -MF \"${_NBL_DXC_CT_CURRENT_DEPENDS_FILEPATH_}\" \"${_SHADER_FILE_PATH_}\"")
	file(WRITE "${_NBL_DXC_CT_CURRENT_IDC_FILEPATH_}" "${_NBL_DXC_CT_WRAPPER_IDC_COMMAND_}")
	
	# create a file contating paths to all input file's dependencies scanned recursively
	if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
		execute_process(COMMAND "${_NBL_DXC_CT_CURRENT_IDC_FILEPATH_}")
	else()
		execute_process(COMMAND "${_NBL_DXC_CT_CURRENT_IDC_FILEPATH_}"
			OUTPUT_QUIET
			ERROR_QUIET
		)
	endif()
	
	file(READ "${_NBL_DXC_CT_CURRENT_DEPENDS_FILEPATH_}" _NBL_DXC_CT_DEPENDENCIES_)
	unset(_NBL_DXC_CT_WRAPPER_IDC_COMMAND_)
	
	# filter DXC output MF's file dependencies and put them into CMake list
	string(REPLACE ": " ";" _NBL_DXC_CT_DEPENDENCIES_S_ "${_NBL_DXC_CT_DEPENDENCIES_}")
	string(REPLACE " \\" ";" _NBL_DXC_CT_DEPENDENCIES_S_ "${_NBL_DXC_CT_DEPENDENCIES_S_}")
	string(REPLACE "\r" "" _NBL_DXC_CT_DEPENDENCIES_S_ "${_NBL_DXC_CT_DEPENDENCIES_S_}")
	string(REPLACE "\n" "" _NBL_DXC_CT_DEPENDENCIES_S_ "${_NBL_DXC_CT_DEPENDENCIES_S_}")
	string(STRIP "${_NBL_DXC_CT_DEPENDENCIES_S_}" _NBL_DXC_CT_DEPENDENCIES_S_)
	list(APPEND _NBL_DXC_CT_DEPENDENCIES_L_ "${_NBL_DXC_CT_DEPENDENCIES_S_}")
	list(TRANSFORM _NBL_DXC_CT_DEPENDENCIES_L_ STRIP)
	list(REMOVE_DUPLICATES _NBL_DXC_CT_DEPENDENCIES_L_)
	
	set(_NBL_DXC_CT_CURRENT_DEPS_LIST_ ${_NBL_DXC_CT_DEPENDENCIES_L_})
	unset(_NBL_DXC_CT_DEPENDENCIES_)
	unset(_NBL_DXC_CT_DEPENDENCIES_S_)
	unset(_NBL_DXC_CT_DEPENDENCIES_L_)
	
	# compile command
	string(APPEND _NBL_DXC_CT_WRAPPER_COMPILE_COMMAND_ "${_TMP_CLEAN_ENV_}\"${DXC_EXT_EXECUTABLE}\" -HV 2021 -T ${_NBL_PROFILE_} ${_TMP_ENTRY_POINT_ARG_} ${_TMP_ENABLE_SPIRV_GEN_ARG_} -I\"${_NBL_DXC_CT_INCLUDE_ROOT_}\" -Zi -fvk-use-scalar-layout -Fo \"${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}\" \"${_SHADER_FILE_PATH_}\"")
	set(_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/compile.cmd")
	file(WRITE "${_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_}" "${_NBL_DXC_CT_WRAPPER_COMPILE_COMMAND_}")
	
	unset(_NBL_DXC_CT_WRAPPER_COMPILE_COMMAND_)
	unset(_TMP_CLEAN_ENV_)
	unset(_TMP_ENTRY_POINT_ARG_)
	unset(_TMP_ENABLE_SPIRV_GEN_ARG_)
	
	if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
		message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} adding _NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_ = '${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}' as OUTPUT of DXC CT target")
	endif()
	
	set(_NBL_DXC_CT_BIN_OUTPUTS_ 
		${_NBL_DXC_CT_BIN_OUTPUTS_}
		"${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}"
	PARENT_SCOPE)
	
	# create custom command per input file with it's all dependencies listed as DEPENDS
	add_custom_command(
		OUTPUT "${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}"
		COMMAND "${_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_}"
		DEPENDS ${_NBL_DXC_CT_CURRENT_DEPS_LIST_}
		WORKING_DIRECTORY "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}"
		USES_TERMINAL
		VERBATIM
	)
	
	unset(_SHADER_FILE_PATH_)
	unset(_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_)
	unset(_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_)
	unset(_NBL_DXC_CT_CURRENT_DEPS_LIST_)
endfunction()

###############################################################################
# list files you would like to compile with DXC using NBL_DXC_CT_ADD_INPUT_FILE
###############################################################################

NBL_DXC_CT_ADD_INPUT_FILE("nbl/builtin/hlsl/barycentric/utils.hlsl" "lib_6_7" "" OFF)

# create custom targets for which DEPENDS are all of input files for DXC compile test
add_custom_target(HLSL_NABLA_COMPILE_TEST DEPENDS ${_NBL_DXC_CT_BIN_OUTPUTS_}
	COMMENT "${CMAKE_COMMAND}" -E echo "Launching Nabla HLSL compile test..."
)

unset(_NBL_DXC_CT_BIN_OUTPUTS_)
unset(_NBL_DXC_CT_BINARY_BIN_)
unset(_NBL_DXC_CT_INCLUDE_ROOT_)
