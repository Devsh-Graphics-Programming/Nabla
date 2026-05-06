include(${NBL_ROOT_PATH}/cmake/common.cmake)

if (NBL_COMPILE_WITH_CUDA)
	set(NBL_EXT_INTERNAL_INCLUDE_DIR "${NBL_ROOT_PATH}/include/nbl/ext/CUDAInterop")

	set(NBL_EXT_CUDA_INTEROP_H
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CUDAInterop.h
		${NBL_EXT_INTERNAL_INCLUDE_DIR}/CUDAInteropNative.h
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
		""
	)

	target_compile_definitions(${LIB_NAME} PRIVATE _NBL_COMPILE_WITH_CUDA_)
	target_include_directories(${LIB_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
	set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME "ext::CUDAInterop")
	add_library(Nabla::ext::CUDAInterop ALIAS ${LIB_NAME})

	set(NBL_EXT_CUDA_INTEROP_NATIVE_LIB "NblExtCUDA_INTEROP_NATIVE")
	add_library(${NBL_EXT_CUDA_INTEROP_NATIVE_LIB} INTERFACE)
	target_link_libraries(${NBL_EXT_CUDA_INTEROP_NATIVE_LIB} INTERFACE
		$<BUILD_INTERFACE:${LIB_NAME}>
		$<INSTALL_INTERFACE:Nabla::ext::CUDAInterop>
		CUDA::toolkit
	)
	set_target_properties(${NBL_EXT_CUDA_INTEROP_NATIVE_LIB} PROPERTIES EXPORT_NAME "ext::CUDAInteropNative")
	add_library(Nabla::ext::CUDAInteropNative ALIAS ${NBL_EXT_CUDA_INTEROP_NATIVE_LIB})
	set(NBL_EXT_CUDA_INTEROP_NATIVE_LIB "${NBL_EXT_CUDA_INTEROP_NATIVE_LIB}" PARENT_SCOPE)
endif()

add_subdirectory(smoke)
