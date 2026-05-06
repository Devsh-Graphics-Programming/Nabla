cmake_minimum_required(VERSION 3.30)
project(NblExtCUDAInteropSmoke CXX)

option(NBL_CUDA_INTEROP_SMOKE_WITH_NATIVE "Build the CUDA native opt-in smoke from an installed Nabla package." OFF)

if(NOT TARGET Nabla::Nabla)
	set(_NBL_CUDA_INTEROP_SMOKE_COMPONENTS Core CUDAInterop)
	if(NBL_CUDA_INTEROP_SMOKE_WITH_NATIVE)
		list(APPEND _NBL_CUDA_INTEROP_SMOKE_COMPONENTS CUDAInteropNative)
	endif()
	find_package(Nabla REQUIRED CONFIG COMPONENTS ${_NBL_CUDA_INTEROP_SMOKE_COMPONENTS})
endif()

enable_testing()

function(nbl_add_cuda_interop_smoke TARGET_NAME SOURCE_FILE)
	add_executable(${TARGET_NAME} ${SOURCE_FILE})
	target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

	add_test(NAME ${TARGET_NAME} COMMAND $<TARGET_FILE:${TARGET_NAME}>)
endfunction()

nbl_add_cuda_interop_smoke(NblExtCUDAInteropPublicBoundarySmoke public_boundary.cpp)
target_link_libraries(NblExtCUDAInteropPublicBoundarySmoke PRIVATE Nabla::Nabla)

if(TARGET Nabla::ext::CUDAInterop)
	nbl_add_cuda_interop_smoke(NblExtCUDAInteropCleanOptInSmoke clean_opt_in.cpp)
	target_link_libraries(NblExtCUDAInteropCleanOptInSmoke PRIVATE Nabla::ext::CUDAInterop)
endif()

if(TARGET Nabla::ext::CUDAInteropNative)
	nbl_add_cuda_interop_smoke(NblExtCUDAInteropNativeOptInSmoke native_opt_in.cpp)
	target_link_libraries(NblExtCUDAInteropNativeOptInSmoke PRIVATE Nabla::ext::CUDAInteropNative)
endif()
