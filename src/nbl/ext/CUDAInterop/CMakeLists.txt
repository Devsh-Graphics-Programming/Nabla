include(${NBL_ROOT_PATH}/cmake/common.cmake)

if (NBL_COMPILE_WITH_CUDA)
	set(NBL_EXT_CUDA_INTEROP_LIB "NblExtCUDA_INTEROP")

	add_library(${NBL_EXT_CUDA_INTEROP_LIB} INTERFACE)
	target_link_libraries(${NBL_EXT_CUDA_INTEROP_LIB} INTERFACE
		$<BUILD_INTERFACE:Nabla>
		$<BUILD_INTERFACE:CUDA::toolkit>
		$<INSTALL_INTERFACE:Nabla::Nabla>
	)
	set_target_properties(${NBL_EXT_CUDA_INTEROP_LIB} PROPERTIES EXPORT_NAME "ext::CUDAInterop")
	add_library(Nabla::ext::CUDAInterop ALIAS ${NBL_EXT_CUDA_INTEROP_LIB})
	set(NBL_EXT_CUDA_INTEROP_LIB "${NBL_EXT_CUDA_INTEROP_LIB}" PARENT_SCOPE)
endif()

add_subdirectory(smoke)
