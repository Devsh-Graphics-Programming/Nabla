# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(../cmake/common.cmake)

option(NBL_FORCE_RELEASE_3RDPARTY "Force map 3rdaprty's configuration regardless Nabla configuration to Release" OFF)
option(NBL_FORCE_RELWITHDEBINFO_3RDPARTY "Force map 3rdaprty's configuration regardless Nabla configuration to RelWithDebInfo" OFF)

# nlohmann_json
set(JSON_BuildTests OFF)
add_subdirectory(nlohmann_json nlohmann_json EXCLUDE_FROM_ALL)

# simdjson
set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
set(_OLD_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS OFF)
set(BUILD_TESTING OFF)
add_subdirectory(simdjson simdjson EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
set(BUILD_TESTING ${_OLD_BUILD_TESTING})
set_property(TARGET simdjson PROPERTY CXX_STANDARD 17)

# DXC
get_filename_component(DXC_SPIRV_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dxc/dxc/external/SPIRV-Tools" ABSOLUTE)
get_filename_component(DXC_SPIRV_HEADERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dxc/dxc/external/SPIRV-Headers" ABSOLUTE)

set(SPIRV-Headers_SOURCE_DIR "${DXC_SPIRV_HEADERS_DIR}" CACHE INTERNAL "")
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")
elseif(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()
option(_NBL_ENABLE_DXC_COMPILE_TESTS_ "Enable it in order to test compilation of Nabla shaders at build time" OFF)
option(_NBL_DXC_COMPILE_TESTS_ENABLE_CMAKE_LOG_ "Enable more verbose log when creating DXC compile test target - only for Nabla developers!" OFF)
set(_NBL_DXC_CT_PREFIX_CMAKE_LOG_DEBUG_ "[DXC_CT]:" CACHE INTERNAL "only for Nabla developers")
add_subdirectory(dxc dxc EXCLUDE_FROM_ALL)

if(_NBL_ENABLE_DXC_COMPILE_TESTS_)
	message(STATUS "_NBL_ENABLE_DXC_COMPILE_TESTS_ enabled, DXC compile test target will be executed at build time!")
endif()

# volk
set(VOLK_PULL_IN_VULKAN OFF CACHE INTERNAL "" FORCE) # Needed to make sure vulkan library will not be linked in examples
	
if (WIN32)
	set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()
add_subdirectory(volk volk EXCLUDE_FROM_ALL)
	
target_include_directories(volk PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include")
target_include_directories(volk_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include")

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_subdirectory(pstl pstl EXCLUDE_FROM_ALL)
endif()

# boost
add_subdirectory(boost boost EXCLUDE_FROM_ALL)

# zlib (target is zlibstatic)
set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(_OLD_SKIP_INSTALL_ALL ${SKIP_INSTALL_ALL})
set(BUILD_SHARED_LIBS OFF)
set(SKIP_INSTALL_ALL ON)

add_subdirectory(zlib zlib EXCLUDE_FROM_ALL)
file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/zlib" DIRECTORY GUARD PROCESS RESULT_VARIABLE NBL_LOCK TIMEOUT 60)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/zlib/zconf.h.included")
	execute_process(COMMAND "${CMAKE_COMMAND}" -E rename zconf.h.included zconf.h 
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/zlib"
	) # clean zlib git working-tree
endif()
file(LOCK "${CMAKE_CURRENT_SOURCE_DIR}/zlib" DIRECTORY RELEASE RESULT_VARIABLE NBL_LOCK)
set(SKIP_INSTALL_ALL ${_OLD_SKIP_INSTALL_ALL})
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})

get_target_property(ZLIB_SOURCE_DIR zlibstatic "SOURCE_DIR")
get_target_property(ZLIB_BINARY_DIR zlibstatic "BINARY_DIR")
get_filename_component(_ORIGINAL_SOURCE_ "${ZLIB_SOURCE_DIR}" ABSOLUTE)
get_filename_component(_COPY_BIN_SOURCE_ "${ZLIB_BINARY_DIR}/copy_source" ABSOLUTE)

add_custom_target(zlib_copy
	COMMAND ${CMAKE_COMMAND} -E rm -Rf "${_COPY_BIN_SOURCE_}"
	COMMAND ${CMAKE_COMMAND} -E copy_directory "${_ORIGINAL_SOURCE_}" "${_COPY_BIN_SOURCE_}"
	COMMAND ${CMAKE_COMMAND} -E rm -f ${_COPY_BIN_SOURCE_}/zconf.h # no duplicates in include search paths!
	COMMENT "Copying zlib..."
)

unset(_ORIGINAL_SOURCE_)
unset(_COPY_BIN_SOURCE_)

#https://stackoverflow.com/questions/47175912/using-cmake-how-to-stop-the-debug-and-release-subdirectories
set_target_properties(zlibstatic PROPERTIES ARCHIVE_OUTPUT_DIRECTORY $<1:${ZLIB_BINARY_DIR}>)
set_target_properties(zlibstatic PROPERTIES DEBUG_POSTFIX "d")
if (MSVC)
	set(ZLIB_BINARY_DIR_DEBUG "${ZLIB_BINARY_DIR}")
	set(ZLIB_BINARY_DIR_RELEASE "${ZLIB_BINARY_DIR}")
	set(ZLIB_BINARY_DIR_RELWITHDEBINFO "${ZLIB_BINARY_DIR}")
	get_target_property(ZLIB_NAME zlibstatic "NAME")
else()
	set(ZLIB_BINARY_DIR_DEBUG ${ZLIB_BINARY_DIR})
	set(ZLIB_BINARY_DIR_RELEASE ${ZLIB_BINARY_DIR})
	set(ZLIB_BINARY_DIR_RELWITHDEBINFO ${ZLIB_BINARY_DIR})
	get_target_property(ZLIB_NAME zlibstatic "OUTPUT_NAME")
endif()

get_target_property(ZLIB_DEBUG_POSTFIX zlibstatic "DEBUG_POSTFIX")
set(ZLIB_INCLUDE_DIR "${THIRD_PARTY_SOURCE_DIR}/zlib;${THIRD_PARTY_BINARY_DIR}/zlib" CACHE INTERNAL "" FORCE)
set(ZLIB_LIBRARY_DEBUG "${ZLIB_BINARY_DIR_DEBUG}/${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_NAME}${ZLIB_DEBUG_POSTFIX}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE INTERNAL "" FORCE)
set(ZLIB_LIBRARY_RELEASE "${ZLIB_BINARY_DIR_RELEASE}/${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE INTERNAL "" FORCE)
set(ZLIB_LIBRARY_RELWITHDEBINFO "${ZLIB_BINARY_DIR_RELWITHDEBINFO}/${CMAKE_STATIC_LIBRARY_PREFIX}${ZLIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE INTERNAL "" FORCE)
set(ZLIB_LIBRARY # for PNGLIB purposes
		$<$<CONFIG:Debug>:${ZLIB_LIBRARY_DEBUG}>	
		$<$<CONFIG:Release>:${ZLIB_LIBRARY_RELEASE}>
		$<$<CONFIG:RelWithDebInfo>:${ZLIB_LIBRARY_RELWITHDEBINFO}>
CACHE INTERNAL "" FORCE)

set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE INTERNAL "Skip SPIRV-Headers install")
set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE INTERNAL "Skip SPIRV-Headers examples")
add_subdirectory("${DXC_SPIRV_HEADERS_DIR}" SPIRV-Headers EXCLUDE_FROM_ALL)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE INTERNAL "Skip SPIRV-Tools install")
set(SPIRV_SKIP_TESTS ON CACHE INTERNAL "Skip SPIRV-Tools tests")
set(SPIRV_SKIP_EXECUTABLES ON CACHE INTERNAL "Skip SPIRV-Tools executable build")
add_subdirectory("${DXC_SPIRV_TOOLS_DIR}" SPIRV-Tools EXCLUDE_FROM_ALL)
set(SKIP_GLSLANG_INSTALL ON CACHE INTERNAL "Skip glslang install")
set(ENABLE_SPVREMAPPER OFF CACHE INTERNAL "SPVRemapper is a spirv compression experimental library")
set(ENABLE_GLSLANG_BINARIES OFF CACHE INTERNAL "Build glslangValidator and spirv-remap?")
set(BUILD_TESTING OFF CACHE INTERNAL "Enable glslang build testing?")
set_property(TARGET SPIRV-Tools-opt PROPERTY CXX_STANDARD 17)
add_subdirectory(glslang glslang EXCLUDE_FROM_ALL)
# because we're integrating glslang via add_subdirectory(), build_info.h is generated to wrong directory (relative to Nabla root) so i have to add extra include dir
target_include_directories(SPIRV PUBLIC "${GLSLANG_GENERATED_INCLUDEDIR}")

set(SHADERC_SKIP_TESTS ON CACHE INTERNAL "Skip shaderc tests?")
set(SHADERC_SKIP_INSTALL ON CACHE INTERNAL "Install shaderc?")

# if it doesn't work without the `touch` on Linux, then fetch the latest submodule head of shaderc and try again
# https://github.com/google/shaderc/issues/568
if (UNIX)
	file(WRITE ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/libshaderc_combined.a "")
endif()
add_subdirectory(shaderc shaderc EXCLUDE_FROM_ALL)

# libjpeg-turbo
option(WITH_SIMD "libjpeg-turbo" ON)
option(WITH_MEM_SRCDST "libjpeg-turbo" ON)
option(WITH_TURBOJPEG "libjpeg-turbo" ON)
option(ENABLE_STATIC "libjpeg-turbo" ON)
option(ENABLE_SHARED "libjpeg-turbo" OFF)
option(WITH_CRT_DLL "libjpeg-turbo" OFF)
option(WITH_FUZZ "libjpeg-turbo" OFF)
option(WITH_JAVA "libjpeg-turbo" OFF)
set(CMAKE_INSTALL_DOCDIR "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/install/docdir")
set(CMAKE_INSTALL_MANDIR "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/install/mandir")
add_subdirectory(libjpeg-turbo libjpeg-turbo EXCLUDE_FROM_ALL)
unset(CMAKE_INSTALL_DOCDIR)
unset(CMAKE_INSTALL_MANDIR)

# blake3
add_subdirectory(blake/c EXCLUDE_FROM_ALL)

#manipulating libpng cmake options from within cmake script
set(PNG_BUILD_ZLIB ON CACHE INTERNAL "" FORCE) # get ZLIB from our /3rdparty
option(PNG_SHARED "Build shared lib" OFF)
option(PNG_TESTS  "Build libpng tests" OFF)
add_subdirectory(libpng libpng EXCLUDE_FROM_ALL)
add_dependencies(png_static zlibstatic)


# OpenEXR
option(_NBL_COMPILE_WITH_OPEN_EXR_ "Build with OpenEXR library" ON)

# Note we are adding it anyway regardless _NBL_COMPILE_WITH_OPEN_EXR_, its because we need Half dependecy 
# it generates on fly in its build directory which doesnt have to be linked (class with inline methods), 
# here is where _NBL_COMPILE_WITH_OPEN_EXR_ plays a role - when disabled OpenEXR won't be built and linked
# but it's core 

set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
set(_OLD_BUILD_TESTING ${BUILD_TESTING})
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS OFF)
set(BUILD_TESTING OFF)
set(PYILMBASE_ENABLE OFF CACHE STRING "" FORCE)
set(OPENEXR_BUILD_UTILS OFF CACHE STRING "" FORCE)
set(OPENEXR_FORCE_INTERNAL_IMATH ON CACHE STRING "" FORCE) # TODO: make it a submodule and force using it maybe instead of letting OpenEXR download it from it's github repository
set(OPENEXR_BUILD_TOOLS OFF CACHE STRING "" FORCE) 
set(OPENEXR_INSTALL OFF CACHE STRING "" FORCE)
set(OPENEXR_INSTALL_DOCS OFF CACHE STRING "" FORCE)
set(OPENEXR_INSTALL_EXAMPLES OFF CACHE STRING "" FORCE)
set(OPENEXR_INSTALL_PKG_CONFIG OFF CACHE STRING "" FORCE)
set(OPENEXR_INSTALL_TOOLS OFF CACHE STRING "" FORCE)

add_subdirectory(openexr openexr EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
set(BUILD_TESTING ${_OLD_BUILD_TESTING})

#gli
option(_NBL_COMPILE_WITH_GLI_ "Build with GLI library" ON)
if(_NBL_COMPILE_WITH_GLI_)
	set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
	set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
	set(_OLD_BUILD_TESTING ${BUILD_TESTING})
	set(BUILD_SHARED_LIBS OFF)
	set(BUILD_STATIC_LIBS OFF)
	set(BUILD_TESTING OFF)
	add_subdirectory(gli gli EXCLUDE_FROM_ALL)
	set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
	set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
	set(BUILD_TESTING ${_OLD_BUILD_TESTING})
endif()

add_library(lzma OBJECT
	lzma/C/Alloc.c
	lzma/C/LzFind.c
	lzma/C/LzmaDec.c
	lzma/C/LzmaEnc.c
)
target_compile_definitions(lzma PRIVATE _7ZIP_ST)


add_library(lz4 OBJECT
	lz4/lib/lz4.c
	lz4/lib/lz4frame.c
	lz4/lib/lz4hc.c
	lz4/lib/xxhash.c
)


add_library(bzip2 OBJECT
	bzip2/blocksort.c
	bzip2/bzlib.c
	bzip2/compress.c
	bzip2/crctable.c
	bzip2/decompress.c
	bzip2/huffman.c
	bzip2/randtable.c
)

add_library(spirv_cross OBJECT
	nbl_spirv_cross/spirv_cfg.cpp
	nbl_spirv_cross/spirv_cross.cpp
	nbl_spirv_cross/spirv_glsl.cpp
	nbl_spirv_cross/spirv_reflect.cpp
	nbl_spirv_cross/spirv_parser.cpp
	nbl_spirv_cross/spirv_cross_parsed_ir.cpp
)
target_compile_definitions(spirv_cross PUBLIC SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)

if(NBL_BUILD_IMGUI)
	add_library(imgui STATIC
		"imgui/imconfig.h"
		"imgui/imgui_demo.cpp"
		"imgui/imgui_draw.cpp"
		"imgui/imgui_internal.h"
		"imgui/imgui_tables.cpp"
		"imgui/imgui_widgets.cpp"
		"imgui/imgui.cpp"
		"imgui/imgui.h"
		"imgui/misc/cpp/imgui_stdlib.cpp"
		"imgui/misc/cpp/imgui_stdlib.h"
		"imgui/imstb_rectpack.h"
		"imgui/imstb_textedit.h"
		"imgui/imstb_truetype.h"
	)
	target_include_directories(imgui
		PUBLIC "imgui"
		PUBLIC "imgui/misc/cpp"
		PUBLIC "imgui/backends"
	)

	# ImPlot
	add_library(implot STATIC
		"implot/implot.h"
		"implot/implot_internal.h"
		"implot/implot.cpp"
		"implot/implot_items.cpp"
	)
	target_include_directories(implot
		PUBLIC "imgui"
	)
	target_link_libraries(implot PUBLIC imgui)
	target_compile_definitions(implot PUBLIC IMPLOT_DEBUG IMPLOT_DLL_EXPORT)
	set_property(TARGET implot PROPERTY CXX_STANDARD 11)
	if(MSVC)
		target_compile_options(implot PRIVATE /MT /W4 /WX /arch:AVX2 /fp:fast /permissive-) 
	else()
		target_compile_options(implot PRIVATE -Wall -Wextra -pedantic -Werror -mavx2 -Ofast)
	endif()
endif()

add_library(aesGladman OBJECT
	aesGladman/aes_ni.c
	aesGladman/aescrypt.c
	aesGladman/aeskey.c
	aesGladman/aestab.c
	aesGladman/fileenc.c
	aesGladman/hmac.c
	aesGladman/prng.c
	aesGladman/pwd2key.c
	aesGladman/sha1.c
)

add_subdirectory(argparse argparse EXCLUDE_FROM_ALL)

option(GLM_TEST_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" ON)
option(GLM_TEST_ENABLE "Build unit tests" OFF)
#add_subdirectory(glm EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})

if (NBL_BUILD_MITSUBA_LOADER)
   option(BUILD_tools "EXPAT: build the xmlwf tool for expat library" OFF)
   option(BUILD_examples "EXPAT: build the examples for expat library" OFF)
   option(BUILD_tests "EXPAT: build the tests for expat library" OFF)
   option(BUILD_shared "EXPAT: build a shared expat library" OFF)
   option(BUILD_doc "EXPAT: build man page for xmlwf" OFF)
   option(USE_libbsd "EXPAT: utilize libbsd (for arc4random_buf)" OFF)
   option(INSTALL "EXPAT: install expat files in cmake install target" OFF)
   set(MSVC_USE_STATIC_CRT ON CACHE BOOL "Use /MT flag (static CRT) when compiling in MSVC")
   add_subdirectory(libexpat/expat EXCLUDE_FROM_ALL)

   set(MITSUBA_LOADER_DEPENDENTS
      expat
   )
   set(MITSUBA_LOADER_DEPENDENTS ${MITSUBA_LOADER_DEPENDENTS} PARENT_SCOPE)
   set(MITSUBA_LOADER_DEPENDENT_LIBS
      expat
   )
   set(MITSUBA_LOADER_DEPENDENT_LIBS ${MITSUBA_LOADER_DEPENDENT_LIBS} PARENT_SCOPE)
endif()

if (NBL_BUILD_BULLET)
	set(BULLET_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/bullet3/src
	)

	set(BULLET_INCLUDE_PATH ${BULLET_INCLUDE_PATH} PARENT_SCOPE)
endif()

# Final gather
set(NBL_3RDPARTY_TARGETS
				lzma
				lz4
				aesGladman
				spirv_cross
				png_static
				zlibstatic
				shaderc_util
				shaderc
				jpeg-static
				bzip2
				simdjson
				nlohmann_json
				glslang
				OGLCompiler 
				OSDependent
				MachineIndependent
				GenericCodeGen
				SPIRV 
				SPIRV-Tools-static # SPIRV-Tools-shared in case of SHARED lib
				SPIRV-Tools-opt
				OpenEXRUtil
				OpenEXRCore
				Iex
				IlmThread
				Imath
				blake3
				${NBL_BOOST_TARGETS}
)
if (_NBL_COMPILE_WITH_OPEN_EXR_)
	list(APPEND NBL_3RDPARTY_TARGETS
		OpenEXR
	)
endif()

#TODO insert more targets that are conditionally built
if (NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_3RDPARTY_TARGETS expat)
endif()
if (NBL_BUILD_IMGUI)
	list(APPEND NBL_3RDPARTY_TARGETS imgui)
endif()
if(ENABLE_HLSL)
	list(APPEND NBL_3RDPARTY_TARGETS HLSL)
endif()

foreach(trgt IN LISTS NBL_3RDPARTY_TARGETS)
		if(NBL_DYNAMIC_MSVC_RUNTIME)
			set_property(TARGET ${trgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
		else()
			set_property(TARGET ${trgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
		endif()
		
		if(MSVC AND NBL_SANITIZE_ADDRESS)
			set_property(TARGET ${trgt} PROPERTY COMPILE_OPTIONS /fsanitize=address)
		endif()
		
		get_target_property(NBL_TARGET_TYPE ${trgt} TYPE)
		if(NOT "${NBL_TARGET_TYPE}" STREQUAL "INTERFACE_LIBRARY")
			# maybe explicit global mapping would be better, to discuss
			if(NBL_FORCE_RELEASE_3RDPARTY) # priority over RWDI
				nbl_adjust_flags(TARGET ${trgt} MAP_RELEASE Release MAP_RELWITHDEBINFO Release MAP_DEBUG Release)
			elseif(NBL_FORCE_RELWITHDEBINFO_3RDPARTY)
				nbl_adjust_flags(TARGET ${trgt} MAP_RELEASE RelWithDebInfo MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG RelWithDebInfo)
			else()
				nbl_adjust_flags(TARGET ${trgt} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
			endif()
		endif()
endforeach()

add_custom_target(3rdparty)
add_dependencies(3rdparty ${NBL_3RDPARTY_TARGETS})

set(_NBL_3RDPARTY_TARGETS_ ${NBL_3RDPARTY_TARGETS} PARENT_SCOPE)

nbl_install_dir("${CMAKE_CURRENT_SOURCE_DIR}/parallel-hashmap/parallel_hashmap")