function(_nbl_cuda_interop_collect_runtime_include_dirs _OUT_INCLUDE_DIRS)
	set(_include_dirs ${ARGN})

	if(DEFINED CUDAToolkit_INCLUDE_DIRS AND NOT "${CUDAToolkit_INCLUDE_DIRS}" STREQUAL "")
		list(APPEND _include_dirs ${CUDAToolkit_INCLUDE_DIRS})
	endif()

	if(TARGET CUDA::toolkit)
		get_target_property(_cuda_toolkit_include_dirs CUDA::toolkit INTERFACE_INCLUDE_DIRECTORIES)
		if(_cuda_toolkit_include_dirs AND NOT _cuda_toolkit_include_dirs STREQUAL "NOTFOUND")
			list(APPEND _include_dirs ${_cuda_toolkit_include_dirs})
		endif()
	endif()

	if(_include_dirs)
		list(REMOVE_DUPLICATES _include_dirs)
	endif()

	set(${_OUT_INCLUDE_DIRS} ${_include_dirs} PARENT_SCOPE)
endfunction()

function(_nbl_cuda_interop_make_runtime_paths_json _OUT_CONTENT)
	set(_include_dirs ${ARGN})
	set(_cuda_runtime_include_dir_entries "")

	foreach(_include_dir IN LISTS _include_dirs)
		if("${_include_dir}" STREQUAL "")
			continue()
		endif()

		file(TO_CMAKE_PATH "${_include_dir}" _include_dir_json)
		string(REPLACE "\"" "\\\"" _include_dir_json "${_include_dir_json}")

		list(APPEND _cuda_runtime_include_dir_entries "    \"${_include_dir_json}\"")
	endforeach()

	set(_json_entry_separator [=[
,
]=])
	list(JOIN _cuda_runtime_include_dir_entries "${_json_entry_separator}" _cuda_runtime_include_dirs)

	set(_json [=[
{
  "cudaRuntimeIncludeDirs": [
@_cuda_runtime_include_dirs@
  ]
}
]=])
	string(CONFIGURE "${_json}" _json @ONLY)
	set(${_OUT_CONTENT} "${_json}" PARENT_SCOPE)
endfunction()

function(_nbl_cuda_interop_collect_configs _OUT_CONFIGS)
	if(CMAKE_CONFIGURATION_TYPES)
		set(_configs ${CMAKE_CONFIGURATION_TYPES})
	elseif(CMAKE_BUILD_TYPE)
		set(_configs "${CMAKE_BUILD_TYPE}")
	else()
		set(_configs Debug)
	endif()

	list(REMOVE_DUPLICATES _configs)
	set(${_OUT_CONFIGS} ${_configs} PARENT_SCOPE)
endfunction()

function(_nbl_cuda_interop_collect_target_runtime_jsons TARGET_NAME _OUT_FILES _OVERRIDE_OUTPUT)
	_nbl_cuda_interop_collect_configs(_configs)
	set(_runtime_jsons "")

	if(NOT "${_OVERRIDE_OUTPUT}" STREQUAL "")
		foreach(_config IN LISTS _configs)
			set(_runtime_paths_json "${_OVERRIDE_OUTPUT}")
			string(REPLACE "$<CONFIG>" "${_config}" _runtime_paths_json "${_runtime_paths_json}")
			if(_runtime_paths_json MATCHES "\\$<")
				message(FATAL_ERROR "Nabla: CUDA interop runtime JSON path supports only plain paths or $<CONFIG>.")
			endif()
			cmake_path(IS_ABSOLUTE _runtime_paths_json _is_abs)
			if(NOT _is_abs)
				cmake_path(ABSOLUTE_PATH _runtime_paths_json BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" OUTPUT_VARIABLE _runtime_paths_json)
			endif()
			cmake_path(NORMAL_PATH _runtime_paths_json OUTPUT_VARIABLE _runtime_paths_json)
			list(APPEND _runtime_jsons "${_runtime_paths_json}")
		endforeach()
		list(REMOVE_DUPLICATES _runtime_jsons)
		set(${_OUT_FILES} ${_runtime_jsons} PARENT_SCOPE)
		return()
	endif()

	foreach(_config IN LISTS _configs)
		string(TOUPPER "${_config}" _config_upper)
		get_target_property(_runtime_output_dir "${TARGET_NAME}" "RUNTIME_OUTPUT_DIRECTORY_${_config_upper}")

		if(NOT _runtime_output_dir OR _runtime_output_dir STREQUAL "NOTFOUND")
			get_target_property(_runtime_output_dir "${TARGET_NAME}" RUNTIME_OUTPUT_DIRECTORY)
		endif()
		if((NOT _runtime_output_dir OR _runtime_output_dir STREQUAL "NOTFOUND") AND DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_config_upper})
			set(_runtime_output_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_config_upper}}")
		endif()
		if((NOT _runtime_output_dir OR _runtime_output_dir STREQUAL "NOTFOUND") AND DEFINED CMAKE_RUNTIME_OUTPUT_DIRECTORY)
			set(_runtime_output_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
		endif()
		if(NOT _runtime_output_dir OR _runtime_output_dir STREQUAL "NOTFOUND")
			if(CMAKE_CONFIGURATION_TYPES)
				set(_runtime_output_dir "${CMAKE_CURRENT_BINARY_DIR}/${_config}")
			else()
				set(_runtime_output_dir "${CMAKE_CURRENT_BINARY_DIR}")
			endif()
		endif()

		string(REPLACE "$<CONFIG>" "${_config}" _runtime_output_dir "${_runtime_output_dir}")
		if(_runtime_output_dir MATCHES "\\$<")
			message(FATAL_ERROR "Nabla: nbl_configure_cuda_interop_runtime supports only plain runtime output directories or $<CONFIG>.")
		endif()

		cmake_path(IS_ABSOLUTE _runtime_output_dir _is_abs)
		if(NOT _is_abs)
			cmake_path(ABSOLUTE_PATH _runtime_output_dir BASE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" OUTPUT_VARIABLE _runtime_output_dir)
		endif()
		cmake_path(NORMAL_PATH _runtime_output_dir OUTPUT_VARIABLE _runtime_output_dir)

		list(APPEND _runtime_jsons "${_runtime_output_dir}/nbl_cuda_interop_runtime.json")
	endforeach()

	list(REMOVE_DUPLICATES _runtime_jsons)
	set(${_OUT_FILES} ${_runtime_jsons} PARENT_SCOPE)
endfunction()

function(nbl_configure_cuda_interop_runtime TARGET_NAME)
	cmake_parse_arguments(_NBL_CUDA_INTEROP "" "RUNTIME_JSON" "INCLUDE_DIRS" ${ARGN})

	if(_NBL_CUDA_INTEROP_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "Nabla: unexpected arguments for nbl_configure_cuda_interop_runtime: ${_NBL_CUDA_INTEROP_UNPARSED_ARGUMENTS}")
	endif()

	if(NOT TARGET "${TARGET_NAME}")
		message(FATAL_ERROR "Nabla: target \"${TARGET_NAME}\" does not exist")
	endif()

	_nbl_cuda_interop_collect_runtime_include_dirs(_include_dirs ${_NBL_CUDA_INTEROP_INCLUDE_DIRS})

	_nbl_cuda_interop_make_runtime_paths_json(_runtime_paths_json_content ${_include_dirs})
	_nbl_cuda_interop_collect_target_runtime_jsons("${TARGET_NAME}" _runtime_paths_jsons "${_NBL_CUDA_INTEROP_RUNTIME_JSON}")

	foreach(_runtime_paths_json IN LISTS _runtime_paths_jsons)
		file(GENERATE OUTPUT "${_runtime_paths_json}" CONTENT "${_runtime_paths_json_content}" TARGET "${TARGET_NAME}")
	endforeach()

	set_source_files_properties(${_runtime_paths_jsons} PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
	target_sources("${TARGET_NAME}" PRIVATE ${_runtime_paths_jsons})
endfunction()

function(nbl_target_link_cuda_interop TARGET_NAME)
	set(_args ${ARGN})
	set(_scope PRIVATE)

	if(_args)
		list(GET _args 0 _first_arg)
		if(_first_arg MATCHES "^(PRIVATE|PUBLIC|INTERFACE)$")
			set(_scope "${_first_arg}")
			list(REMOVE_AT _args 0)
		endif()
	endif()

	cmake_parse_arguments(_NBL_CUDA_INTEROP "" "RUNTIME_JSON" "INCLUDE_DIRS" ${_args})

	if(_NBL_CUDA_INTEROP_UNPARSED_ARGUMENTS)
		message(FATAL_ERROR "Nabla: unexpected arguments for nbl_target_link_cuda_interop: ${_NBL_CUDA_INTEROP_UNPARSED_ARGUMENTS}")
	endif()

	if(NOT TARGET "${TARGET_NAME}")
		message(FATAL_ERROR "Nabla: target \"${TARGET_NAME}\" does not exist")
	endif()
	if(NOT TARGET Nabla::ext::CUDAInterop)
		message(FATAL_ERROR "Nabla: Nabla::ext::CUDAInterop is not available. Request the CUDAInterop package component or enable NBL_COMPILE_WITH_CUDA.")
	endif()

	target_link_libraries("${TARGET_NAME}" ${_scope} Nabla::ext::CUDAInterop)
	nbl_configure_cuda_interop_runtime("${TARGET_NAME}"
		RUNTIME_JSON "${_NBL_CUDA_INTEROP_RUNTIME_JSON}"
		INCLUDE_DIRS ${_NBL_CUDA_INTEROP_INCLUDE_DIRS}
	)
endfunction()
