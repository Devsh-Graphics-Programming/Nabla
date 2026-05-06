include(${NBL_ROOT_PATH}/cmake/common.cmake)

if (NBL_COMPILE_WITH_CUDA)
	set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include/nbl/ext/CUDAInterop")

	set(NBL_EXT_CUDA_INTEROP_H
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CUDAInterop.h
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CCUDADevice.h
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CCUDAExportableMemory.h
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CCUDAHandler.h
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CCUDAImportedMemory.h
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CCUDAImportedSemaphore.h
	)

	set(NBL_EXT_CUDA_INTEROP_SRC
		CCUDADevice.cpp
		CCUDAExportableMemory.cpp
		CCUDAHandler.cpp
		CCUDAImportedMemory.cpp
		CCUDAImportedSemaphore.cpp
	)

	nbl_create_ext_library_project(
		CUDA_INTEROP
		"${NBL_EXT_CUDA_INTEROP_H}"
		"${NBL_EXT_CUDA_INTEROP_SRC}"
		""
		""
		"_NBL_COMPILE_WITH_CUDA_"
	)

	target_link_libraries(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:CUDA::toolkit>)
	set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME "ext::CUDAInterop")
	add_library(Nabla::ext::CUDAInterop ALIAS ${LIB_NAME})
endif()

add_subdirectory(smoke)
