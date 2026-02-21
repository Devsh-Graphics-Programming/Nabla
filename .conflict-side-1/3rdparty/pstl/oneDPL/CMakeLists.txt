cmake_minimum_required(VERSION 3.20)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/toolchains/android/build.cmake)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	if(NOT DEFINED ONETBB_INSTALL_DIR)
		message(FATAL_ERROR "oneDPL submodule requires valid ONETBB_INSTALL_DIR!")
	endif()
	
	set(TBB_DIR ${ONETBB_INSTALL_DIR}/lib/cmake/TBB CACHE INTERNAL "" FORCE)
	set(ONEDPL_BACKEND tbb CACHE STRING "" FORCE)
	add_subdirectory(oneDPL oneDPL)

	if(ANDROID)
		add_compile_options(
			$<$<CONFIG:DEBUG>:-glldb>
		)
	endif()
endif()