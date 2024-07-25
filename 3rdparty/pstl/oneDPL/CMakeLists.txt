cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# assume built-in pthreads on MacOS
if(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON CACHE BOOL "")
endif()

if(ANDROID)
	include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/toolchains/android/build.cmake)
endif()

if(NOT DEFINED ONETBB_INSTALL_DIR)
	message(FATAL_ERROR "oneDPL submodule requires valid ONETBB_INSTALL_DIR!")
endif()

set(TBB_DIR ${ONETBB_INSTALL_DIR}/lib/cmake/TBB CACHE INTERNAL "" FORCE)
set(ONEDPL_BACKEND tbb CACHE STRING "" FORCE)
set(CMAKE_FIND_USE_CMAKE_SYSTEM_PATH ON CACHE BOOL "")
set(ONE_DPL_INSTALL_OVERRIDE "${CMAKE_CURRENT_BINARY_DIR}/output" CACHE STRING "" FORCE)
find_package(Threads REQUIRED)

add_subdirectory(oneDPL oneDPL)

if(ANDROID)
	add_compile_options(
		$<$<CONFIG:DEBUG>:-glldb>
	)
endif()
