include(common)
include(NablaCUDAInteropHelpers)

if (NBL_COMPILE_WITH_CUDA)
	set(NBL_EXT_CUDA_INTEROP_LIB "NblExtCUDA_INTEROP")

	file(GLOB NBL_EXT_CUDA_INTEROP_IDE_HEADERS CONFIGURE_DEPENDS "${NBL_ROOT_PATH}/include/nbl/ext/CUDAInterop/*.h")
	set(NBL_EXT_CUDA_INTEROP_IDE_SOURCES
		${NBL_EXT_CUDA_INTEROP_IDE_HEADERS}
		CMakeLists.txt
		README.md
	)
	set_source_files_properties(${NBL_EXT_CUDA_INTEROP_IDE_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)

	# Header-only opt-in target. It builds no artifact and adds CUDA SDK usage requirements only for native interop consumers.
	add_library(${NBL_EXT_CUDA_INTEROP_LIB} INTERFACE EXCLUDE_FROM_ALL ${NBL_EXT_CUDA_INTEROP_IDE_SOURCES})
	target_link_libraries(${NBL_EXT_CUDA_INTEROP_LIB} INTERFACE
		Nabla
		CUDA::toolkit
	)
	set_target_properties(${NBL_EXT_CUDA_INTEROP_LIB} PROPERTIES EXPORT_NAME "ext::CUDAInterop")
	add_library(Nabla::ext::CUDAInterop ALIAS ${NBL_EXT_CUDA_INTEROP_LIB})
	set(NBL_EXT_CUDA_INTEROP_LIB "${NBL_EXT_CUDA_INTEROP_LIB}" PARENT_SCOPE)
endif()

add_subdirectory(smoke)
