############################################################################
# list files you want to compile with DXC in _NBL_ENABLE_DXC_COMPILE_TESTS_
# once we have guarantee all of shaders in Nabla can be compiled with DXC,
# we can simply grep all of those files/use existing vars holding paths to
# shaders and remove manual adding the files bellow
###################################################

set(_NBL_DXC_CT_INCLUDE_ROOT_ "${NBL_ROOT_PATH}/include") # Nabla include root path for shaders
set(_NBL_DXC_CT_BINARY_BIN_ "${CMAKE_CURRENT_BINARY_DIR}") # compile-test directory

if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
	message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} _NBL_DXC_CT_INCLUDE_ROOT_ = '${_NBL_DXC_CT_INCLUDE_ROOT_}'")
	message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} _NBL_DXC_CT_BINARY_BIN_ = '${_NBL_DXC_CT_BINARY_BIN_}'")
endif()

# this is how you add file for compilation test
set(_NBL_DXC_CT_SHADER_FILES_ 
	"${_NBL_DXC_CT_INCLUDE_ROOT_}/nbl/builtin/hlsl/barycentric/utils.hlsl"
)

foreach(_SHADER_FILE_PATH_ IN LISTS _NBL_DXC_CT_SHADER_FILES_)
	if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
		message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} adding _SHADER_FILE_PATH_ = '${_SHADER_FILE_PATH_}' as INPUT of DXC CT target")
	endif()
	
	cmake_path(RELATIVE_PATH _SHADER_FILE_PATH_ BASE_DIRECTORY "${_NBL_DXC_CT_INCLUDE_ROOT_}" OUTPUT_VARIABLE _SHADER_NEW_RELATIVE_FILE_PATH_)
	cmake_path(GET _SHADER_NEW_RELATIVE_FILE_PATH_ STEM LAST_ONLY _SHADER_NEW_STEAM_)
	cmake_path(REMOVE_EXTENSION _SHADER_NEW_RELATIVE_FILE_PATH_ LAST_ONLY OUTPUT_VARIABLE _SHADER_NEW_RELATIVE_OUTPUT_DIR_)
	
	set(_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_ "${_NBL_DXC_CT_BINARY_BIN_}/${_SHADER_NEW_RELATIVE_OUTPUT_DIR_}/bin")
	set(_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/${_SHADER_NEW_STEAM_}.bin")
	
	# MSVC and other generators have "SetEnv" task - we need to make sure our enviroment is clean while invoking DXC compilation
	string(APPEND _TMP_CLEAN_ENV_ "set PATH=$ENV{PATH}\nset CAExcludePath=\nset LIB=\nset LIBPATH=\nset INCLUDE=\nset EXTERNAL_INCLUDE=\n")
	
	# include-dependency input file chain
	set(_NBL_DXC_CT_CURRENT_DEPENDS_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/depends.txt")
	set(_NBL_DXC_CT_CURRENT_IDC_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/depends.cmd")
	string(APPEND _NBL_DXC_CT_WRAPPER_IDC_COMMAND_ "${_TMP_CLEAN_ENV_}\"${DXC_EXECUTABLE}\" -HV 2021 -T lib_6_7 -I\"${_NBL_DXC_CT_INCLUDE_ROOT_}\" -Zi -MF \"${_NBL_DXC_CT_CURRENT_DEPENDS_FILEPATH_}\" \"${_SHADER_FILE_PATH_}\"")
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
	string(APPEND _NBL_DXC_CT_WRAPPER_COMPILE_COMMAND_ "${_TMP_CLEAN_ENV_}\"${DXC_EXECUTABLE}\" -HV 2021 -T lib_6_7 -I\"${_NBL_DXC_CT_INCLUDE_ROOT_}\" -Zi -Fo \"${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}\" \"${_SHADER_FILE_PATH_}\"")
	set(_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_ "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}/compile.cmd")
	file(WRITE "${_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_}" "${_NBL_DXC_CT_WRAPPER_COMPILE_COMMAND_}")
	
	unset(_NBL_DXC_CT_WRAPPER_COMPILE_COMMAND_)
	unset(_TMP_CLEAN_ENV_)
	
	if(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_)
		message("${_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_} adding _NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_ = '${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}' as OUTPUT of DXC CT target")
	endif()
	
	list(APPEND _NBL_DXC_CT_BIN_OUTPUTS_ "${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}")
	
	# create custom command per input file with it's all dependencies listed as DEPENDS
	add_custom_command(
		OUTPUT "${_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_}"
		COMMAND "${_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_}"
		DEPENDS ${_NBL_DXC_CT_CURRENT_DEPS_LIST_}
		WORKING_DIRECTORY "${_NBL_DXC_CT_CURRENT_OUTPUT_BIN_DIR_}"
		USES_TERMINAL
		VERBATIM
	)
	
	unset(_NBL_DXC_CT_CURRENT_OUTPUT_FILEPATH_)
	unset(_NBL_DXC_CT_CURRENT_SHADER_COMPILE_FILEPATH_)
	unset(_NBL_DXC_CT_CURRENT_DEPS_LIST_)
endforeach()

# create custom targets for which DEPENDS are all of input files for DXC compile test
add_custom_target(HLSL_NABLA_COMPILE_TEST DEPENDS ${_NBL_DXC_CT_BIN_OUTPUTS_}
	COMMENT "${CMAKE_COMMAND}" -E echo "Launching Nabla HLSL compile test..."
)

add_dependencies(HLSL_NABLA_COMPILE_TEST dxc)

unset(_NBL_DXC_CT_BIN_OUTPUTS_)
unset(_NBL_DXC_CT_BINARY_BIN_)
unset(_NBL_DXC_CT_SHADER_FILES_)
unset(_NBL_DXC_CT_INCLUDE_ROOT_)