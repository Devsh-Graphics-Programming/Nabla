cmake_minimum_required(VERSION 3.30)
project(NblExtCUDAInteropSmoke CXX)

if(NOT TARGET Nabla::Nabla)
	find_package(Nabla REQUIRED CONFIG COMPONENTS Core CUDAInterop)
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
	nbl_add_cuda_interop_smoke(NblExtCUDAInteropOptInSmoke opt_in.cpp)
	target_link_libraries(NblExtCUDAInteropOptInSmoke PRIVATE Nabla::ext::CUDAInterop)
endif()
