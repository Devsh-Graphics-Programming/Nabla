if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	cmake_minimum_required(VERSION 3.30)
	project(NblExtCUDAInteropSmoke CXX)
	find_package(Nabla REQUIRED CONFIG)
endif()

enable_testing()

function(nbl_add_cuda_interop_smoke TARGET_NAME SOURCE_FILE)
	add_executable(${TARGET_NAME} ${SOURCE_FILE})
	target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
	if(MSVC)
		target_compile_options(${TARGET_NAME} PRIVATE
			/Gm-
			/bigobj
			/Zc:wchar_t
			/Zc:preprocessor
			/Zc:inline
			/Zc:forScope
		)
	endif()
	if(COMMAND nbl_adjust_flags)
		nbl_adjust_flags(TARGET ${TARGET_NAME} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
	endif()

	set(_NBL_CUDA_INTEROP_SMOKE_PATH_MODS "PATH=path_list_prepend:$<TARGET_FILE_DIR:Nabla::Nabla>")
	if(CUDAToolkit_BIN_DIR)
		list(APPEND _NBL_CUDA_INTEROP_SMOKE_PATH_MODS "PATH=path_list_prepend:${CUDAToolkit_BIN_DIR}")
	endif()

	add_test(NAME ${TARGET_NAME} COMMAND $<TARGET_FILE:${TARGET_NAME}>)
	set_tests_properties(${TARGET_NAME} PROPERTIES
		ENVIRONMENT_MODIFICATION "${_NBL_CUDA_INTEROP_SMOKE_PATH_MODS}"
	)
endfunction()

nbl_add_cuda_interop_smoke(NblExtCUDAInteropPublicBoundarySmoke public_boundary.cpp)
target_link_libraries(NblExtCUDAInteropPublicBoundarySmoke PRIVATE Nabla::Nabla)

if(TARGET Nabla::ext::CUDAInterop)
	nbl_add_cuda_interop_smoke(NblExtCUDAInteropOptInSmoke opt_in.cpp)
	target_link_libraries(NblExtCUDAInteropOptInSmoke PRIVATE Nabla::ext::CUDAInterop)
endif()
