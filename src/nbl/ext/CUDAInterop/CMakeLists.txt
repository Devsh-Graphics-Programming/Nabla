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

	set(NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIRS)
	if(CUDAToolkit_ROOT)
		list(APPEND NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIRS
			"${CUDAToolkit_ROOT}/include"
			"${CUDAToolkit_ROOT}/include/cccl"
		)
	endif()
	list(APPEND NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
	list(REMOVE_DUPLICATES NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIRS)

	foreach(_NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIR IN LISTS NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIRS)
		target_include_directories(${LIB_NAME} BEFORE PUBLIC $<BUILD_INTERFACE:${_NBL_EXT_CUDA_INTEROP_CUDA_INCLUDE_DIR}>)
	endforeach()
	set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME "ext::CUDAInterop")
	add_library(Nabla::ext::CUDAInterop ALIAS ${LIB_NAME})
endif()

add_subdirectory(smoke)
