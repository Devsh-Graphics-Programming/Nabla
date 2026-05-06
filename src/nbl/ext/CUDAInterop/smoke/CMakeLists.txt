enable_testing()

set(_NBL_CUDA_INTEROP_SMOKE_CONFIG_ARGS)
if(CMAKE_CONFIGURATION_TYPES)
	set(_NBL_CUDA_INTEROP_SMOKE_CONFIG_ARGS --config $<CONFIG>)
endif()

function(nbl_add_cuda_interop_smoke TARGET_NAME SOURCE_FILE)
	add_executable(${TARGET_NAME} EXCLUDE_FROM_ALL ${SOURCE_FILE})
	target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
	nbl_adjust_flags(TARGET ${TARGET_NAME} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)

	set(_NBL_CUDA_INTEROP_SMOKE_PATH_MODS "PATH=path_list_prepend:$<TARGET_FILE_DIR:Nabla::Nabla>")
	if(CUDAToolkit_BIN_DIR)
		list(APPEND _NBL_CUDA_INTEROP_SMOKE_PATH_MODS "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
	endif()

	add_test(
		NAME ${TARGET_NAME}.build
		COMMAND ${CMAKE_COMMAND} --build "${CMAKE_CURRENT_BINARY_DIR}" --target ${TARGET_NAME} ${_NBL_CUDA_INTEROP_SMOKE_CONFIG_ARGS}
	)
	add_test(NAME ${TARGET_NAME}.run COMMAND $<TARGET_FILE:${TARGET_NAME}>)
	set_tests_properties(${TARGET_NAME}.run PROPERTIES
		DEPENDS ${TARGET_NAME}.build
		ENVIRONMENT_MODIFICATION "${_NBL_CUDA_INTEROP_SMOKE_PATH_MODS}"
	)
endfunction()

nbl_add_cuda_interop_smoke(NblExtCUDAInteropPublicBoundarySmoke public_boundary.cpp)
target_link_libraries(NblExtCUDAInteropPublicBoundarySmoke PRIVATE Nabla::Nabla)

if(TARGET Nabla::ext::CUDAInterop)
	nbl_add_cuda_interop_smoke(NblExtCUDAInteropOptInSmoke opt_in.cpp)
	target_link_libraries(NblExtCUDAInteropOptInSmoke PRIVATE Nabla::ext::CUDAInterop)
endif()
