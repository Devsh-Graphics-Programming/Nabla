function(nbl_target_link_cuda_interop TARGET_NAME SCOPE)
	if(NOT SCOPE MATCHES "^(PRIVATE|PUBLIC|INTERFACE)$")
		set(SCOPE PRIVATE)
	endif()
	cmake_parse_arguments(_NBL_CUDA_INTEROP "" "RUNTIME_JSON" "INCLUDE_DIRS" ${ARGN})
	target_link_libraries("${TARGET_NAME}" ${SCOPE} Nabla::ext::CUDAInterop)
	set(_include_dir_entries "")
	foreach(_include_dir IN LISTS _NBL_CUDA_INTEROP_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
		if(_include_dir)
			file(TO_CMAKE_PATH "${_include_dir}" _include_dir)
			list(APPEND _include_dir_entries "    \"${_include_dir}\"")
		endif()
	endforeach()
	list(JOIN _include_dir_entries "," _include_dirs_json)
	set(_runtime_json [=[
{
  "cudaRuntimeIncludeDirs": [
@_include_dirs_json@
  ]
}
]=])
	string(CONFIGURE "${_runtime_json}" _runtime_json @ONLY)
	set(_runtime_json_path "$<TARGET_FILE_DIR:${TARGET_NAME}>/nbl_cuda_interop_runtime.json")
	if(_NBL_CUDA_INTEROP_RUNTIME_JSON)
		set(_runtime_json_path "${_NBL_CUDA_INTEROP_RUNTIME_JSON}")
	endif()
	file(GENERATE OUTPUT "${_runtime_json_path}" CONTENT "${_runtime_json}" TARGET "${TARGET_NAME}")
endfunction()
