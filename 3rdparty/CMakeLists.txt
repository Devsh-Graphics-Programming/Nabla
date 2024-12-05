# Copyright (C) 2018-2024 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

cmake_minimum_required(VERSION 3.29)
include(../cmake/init/buildSystem.cmake NO_POLICY_SCOPE) 

option(NBL_3RDPARTY_ENABLE_FOLDER_GROUPING "Group 3rdparty into folders, useful for IDE but slows down a bit generate time" ON)
option(NBL_3RDPARTY_EXPORT_TO_BUILD_TREE "Export all targets to file as 3rdparty & extensions interface targets" ON)
option(NBL_FORCE_RELEASE_3RDPARTY "Force map 3rdaprty's configuration regardless Nabla configuration to Release" OFF)
option(NBL_FORCE_RELWITHDEBINFO_3RDPARTY "Force map 3rdaprty's configuration regardless Nabla configuration to RelWithDebInfo" OFF)

project(3rdparty LANGUAGES CXX C ASM)

# 3rdparty
option(NBL_COMPILE_WITH_CUDA "[Nabla]: Compile with CUDA interop" OFF)
option(_NBL_COMPILE_WITH_OPEN_EXR_ "[Nabla]: Build with OpenEXR library" ON)
option(_NBL_COMPILE_WITH_GLI_ "[Nabla]: Build with GLI library" ON)
option(NBL_BUILD_DPL "[Nabla]: Enable DPL (Dynamic Parallelism Library)" OFF) # TODO: rename to COMPILE_WITH & update references

# extensions
option(NBL_BUILD_IMGUI "[EXT]: Build with IMGUI libraries, enable nbl::ext::ImGui" OFF)
option(NBL_BUILD_MSDF "[EXT]: Build with MSDF library, enable nbl::ext::TextRendering" OFF)
option(NBL_BUILD_MITSUBA_LOADER "[EXT]: Build with expact, enable nbl::ext::MitsubaLoader" OFF) # TODO: once it compies turn this ON by default!
# option(NBL_BUILD_BULLET "[EXT]: Enable Bullet Physics building and integration" OFF) TODO: is this still used anywhere? Check it

include(../cmake/common.cmake)

# we have 2 interface targets shared with Nabla build system
# - "3rdparty" consist of dependencies required to build Nabla library, some of them can be controled with NBL_BUILD_WITH_* options
# - "extensions" are dependencies which are not compiled as part of Nabla library itself but they can be used to create Nabla extensions
# NBL_3RDPARTY_EXPORT_TO_BUILD_TREE controls if targets available as part of those 2 interfaces are exported to file as "imported" targets

add_library(3rdparty INTERFACE)
add_library(extensions INTERFACE)

set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)
set(SKIP_INSTALL_ALL ON)

# Vulkan SDK
find_package(Vulkan)
if (Vulkan_FOUND)
	message(STATUS "Building with Vulkan driver")
else()
	message(STATUS "Vulkan driver is not enabled")
endif()

# Vulkan-Headers
add_library(Vulkan-Headers INTERFACE) # do not add subdirectory, messed target there
target_include_directories(Vulkan-Headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include")

# volk
set(VOLK_PULL_IN_VULKAN OFF) # need to make sure vulkan library will not be linked in examples
	
if (WIN32)
	set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
endif()

add_subdirectory(volk EXCLUDE_FROM_ALL)
target_include_directories(volk PUBLIC $<TARGET_PROPERTY:Vulkan-Headers,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(volk_headers INTERFACE $<TARGET_PROPERTY:Vulkan-Headers,INTERFACE_INCLUDE_DIRECTORIES>)

# CUDA
if(NBL_COMPILE_WITH_CUDA)
	find_package(CUDAToolkit REQUIRED)
	if(${CUDAToolkit_VERSION} VERSION_GREATER "9.0")
		message(STATUS "CUDA version 9.0+ found!")
	else()
		message(FATAL_ERROR "CUDA version 9.0+ needed for C++14 support!")
	endif()
	
	set(CMAKE_CUDA_STANDARD 14 CACHE INTERNAL "")
	set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE INTERNAL "")
	
	message(STATUS "Building with CUDA interop")
	
	if(NBL_BUILD_OPTIX)
		find_package(OPTIX REQUIRED)
		message(STATUS "CUDA enabled and OptiX found!")
	endif()
	
	set(_NBL_COMPILE_WITH_CUDA_ ${NBL_COMPILE_WITH_CUDA}
	PARENT_SCOPE)
	
	if (NBL_BUILD_OPTIX)
		set(_NBL_BUILD_OPTIX_ ${NBL_BUILD_OPTIX}
		PARENT_SCOPE)
	endif()
else()
	if(NBL_BUILD_OPTIX)
		message(FATAL_ERROR "You cannot build Optix without enabled CUDA! NBL_COMPILE_WITH_CUDA must be ON!")
	endif()
endif()

# TODO: update correctly

if (UNIX)	
	if(NOT ANDROID)
		find_package(X11 REQUIRED)
		set(CMAKE_THREAD_PREFER_PTHREAD 1)
		
		if(NOT APPLE)
			add_library(DisplaySystemProxy INTERFACE)
			target_include_directories(DisplaySystemProxy PUBLIC ${X11_INCLUDE_DIR} ${X11_xf86vmode_INCLUDE_PATH})
		endif()
	endif()
endif()

if (ANDROID)
	add_library(android_native_app_glue STATIC ${ANDROID_NDK_ROOT_PATH}/sources/android/native_app_glue/android_native_app_glue.c)
	target_include_directories(android_native_app_glue PUBLIC "${ANDROID_NDK_ROOT_PATH}/sources/android/native_app_glue")
endif()

# nlohmann_json
set(JSON_BuildTests OFF)
add_subdirectory(nlohmann_json nlohmann_json EXCLUDE_FROM_ALL)

# simdjson
add_subdirectory(simdjson simdjson EXCLUDE_FROM_ALL)
set_target_properties(simdjson PROPERTIES CXX_STANDARD 17)

# freetype2
set(FT_WITH_HARFBUZZ OFF)
add_subdirectory(freetype2 EXCLUDE_FROM_ALL)
add_library(Freetype::Freetype ALIAS freetype) # alias target for MSDFGEN not trying to find freetype system-wide but rather to use the logical target available in our build tree

# msdfgen
set(MSDFGEN_USE_VCPKG OFF)
set(MSDFGEN_USE_SKIA OFF)
set(MSDFGEN_CORE_ONLY OFF)
set(MSDFGEN_INSTALL OFF)
set(MSDFGEN_BUILD_STANDALONE OFF)
set(MSDFGEN_USE_CPP11 ON)
set(MSDFGEN_DISABLE_SVG ON)
set(MSDFGEN_DISABLE_PNG ON)
add_subdirectory(msdfgen msdfgen EXCLUDE_FROM_ALL)

if(MSDFGEN_CORE_ONLY)
	message(FATAL_ERROR "Internal error, MSDF must be used with extension target as well!")
else()
	set(NBL_MSDFGEN_TARGETS msdfgen-ext msdfgen-core)
endif()

# DXC
add_subdirectory(dxc dxc EXCLUDE_FROM_ALL)

# glslang
set(SKIP_GLSLANG_INSTALL ON)
set(ENABLE_SPVREMAPPER OFF)
set(ENABLE_GLSLANG_BINARIES OFF)
add_subdirectory(glslang EXCLUDE_FROM_ALL)
target_include_directories(SPIRV PUBLIC "${GLSLANG_GENERATED_INCLUDEDIR}") # because we're integrating glslang via add_subdirectory(), build_info.h is generated to wrong directory (relative to Nabla root) so i have to add extra include dir: TODO CHECK IT

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_subdirectory(pstl EXCLUDE_FROM_ALL) # TODO: check targets when dealing with UNIX builds
endif()

# boost
add_subdirectory(boost boost EXCLUDE_FROM_ALL)

# TODO: update this + PNG target to latest and refactor following code chunk
######################################## <-

# zlib TODO: update it to latest
add_subdirectory(zlib EXCLUDE_FROM_ALL)
get_target_property(ZLIB_BINARY_DIR zlibstatic "BINARY_DIR")
set_target_properties(zlibstatic PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ZLIB_BINARY_DIR})
set_target_properties(zlibstatic PROPERTIES DEBUG_POSTFIX "d")

set(ZLIB_INCLUDE_DIR $<TARGET_PROPERTY:zlibstatic,INCLUDE_DIRECTORIES>) # do not remove
set(ZLIB_LIBRARY $<TARGET_LINKER_FILE:zlibstatic>) # do not remove

add_library(zlibConfig INTERFACE)
target_include_directories(zlibConfig INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/zlib")

# png
set(PNG_BUILD_ZLIB ON)
set(PNG_SHARED OFF)
set(PNG_TESTS OFF)
add_subdirectory(libpng EXCLUDE_FROM_ALL)

add_library(pngConfig INTERFACE)
target_include_directories(pngConfig 
	INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/libpng"
	INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/libpng"
)

add_dependencies(png_static zlibstatic)
##################################### <-

# if it doesn't work without the `touch` on Linux, then fetch the latest submodule head of shaderc and try again
# https://github.com/google/shaderc/issues/568
if (UNIX)
	file(WRITE ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/libshaderc_combined.a "") # W T F, TODO take closer look at it when dealing with UNIX builds
endif()
set(SHADERC_SKIP_TESTS ON)
set(SHADERC_SKIP_INSTALL ON)
add_subdirectory(shaderc EXCLUDE_FROM_ALL)

# libjpeg-turbo
set(WITH_SIMD ON)
set(WITH_MEM_SRCDST ON)
set(WITH_TURBOJPEG ON)
set(ENABLE_STATIC ON)
set(ENABLE_SHARED OFF)
set(WITH_CRT_DLL OFF)
set(WITH_FUZZ OFF)
set(WITH_JAVA OFF)
set(CMAKE_INSTALL_DOCDIR "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/install/docdir")
set(CMAKE_INSTALL_MANDIR "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo/install/mandir")
add_subdirectory(libjpeg-turbo EXCLUDE_FROM_ALL) # BY THE WAY WE NEVER USED THE TURBO VERSION WITH SIMD BUT PURE JPEG XD TODO: USE TUURBO
add_library(jpegConfig INTERFACE)
target_include_directories(jpegConfig
	INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo"
	INTERFACE "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo"
)

unset(CMAKE_INSTALL_DOCDIR)
unset(CMAKE_INSTALL_MANDIR)

# blake3
add_subdirectory(blake/c EXCLUDE_FROM_ALL)

# Imath
set(IMATH_INSTALL OFF)
add_subdirectory(imath EXCLUDE_FROM_ALL)

# OpenEXR
option(_NBL_COMPILE_WITH_OPEN_EXR_ "Build with OpenEXR library" ON)

if(_NBL_COMPILE_WITH_OPEN_EXR_)
	# Deflate
	set(LIBDEFLATE_BUILD_SHARED_LIB OFF)
	set(LIBDEFLATE_BUILD_STATIC_LIB ON)
	set(LIBDEFLATE_GZIP_SUPPORT OFF)
	set(LIBDEFLATE_BUILD_GZIP OFF)
	add_subdirectory(libdeflate EXCLUDE_FROM_ALL)
	set(libdeflate_DIR "${CMAKE_CURRENT_BINARY_DIR}/libdeflate")

	# OpenEXR
	set(OPENEXR_FORCE_INTERNAL_DEFLATE ON) # trick it into thinking its internal
	set(EXR_DEFLATE_LIB libdeflate_static) # and pass deflate target directly from our build tree
	set(OPENEXR_FORCE_INTERNAL_IMATH ON) # similar case, force it to look into target from build tree
	set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
	set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
	set(_OLD_BUILD_TESTING ${BUILD_TESTING})
	set(BUILD_SHARED_LIBS OFF)
	set(BUILD_STATIC_LIBS OFF)
	set(BUILD_TESTING OFF)
	set(PYILMBASE_ENABLE OFF)
	set(OPENEXR_BUILD_UTILS OFF)
	set(OPENEXR_BUILD_TOOLS OFF)
	set(OPENEXR_INSTALL OFF)
	set(OPENEXR_INSTALL_DOCS OFF)
	set(OPENEXR_INSTALL_EXAMPLES OFF)
	set(OPENEXR_INSTALL_PKG_CONFIG OFF)
	set(OPENEXR_INSTALL_TOOLS OFF)
	add_subdirectory(openexr EXCLUDE_FROM_ALL)
	set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
	set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
	set(BUILD_TESTING ${_OLD_BUILD_TESTING})
endif()

# gli
if(_NBL_COMPILE_WITH_GLI_)
	add_subdirectory(gli EXCLUDE_FROM_ALL)
endif()

# TODO check for updates + better move to separate file if possible (if CMake provided use its target)
add_library(lzma STATIC
	lzma/C/Alloc.c
	lzma/C/LzFind.c
	lzma/C/LzmaDec.c
	lzma/C/LzmaEnc.c
	lzma/C/LzFind.c
	lzma/C/LzFindMt.c
	lzma/C/LzFindOpt.c
	lzma/C/CpuArch.c
	lzma/C/Threads.c
)
target_compile_definitions(lzma PRIVATE _7ZIP_ST)

# TODO check for updates + better move to separate file if possible (if CMake provided use its target)
add_library(lz4 STATIC
	lz4/lib/lz4.c
	lz4/lib/lz4frame.c
	lz4/lib/lz4hc.c
	lz4/lib/xxhash.c
)

# TODO check for updates + better move to separate file if possible (if CMake provided use its target)
add_library(bzip2 STATIC
	bzip2/blocksort.c
	bzip2/bzlib.c
	bzip2/compress.c
	bzip2/crctable.c
	bzip2/decompress.c
	bzip2/huffman.c
	bzip2/randtable.c
)

# TODO check for updates + better move to separate file if possible (if CMake provided use its target)
add_library(spirv_cross STATIC
	nbl_spirv_cross/spirv_cfg.cpp
	nbl_spirv_cross/spirv_cross.cpp
	nbl_spirv_cross/spirv_glsl.cpp
	nbl_spirv_cross/spirv_reflect.cpp
	nbl_spirv_cross/spirv_parser.cpp
	nbl_spirv_cross/spirv_cross_parsed_ir.cpp
)
target_compile_definitions(spirv_cross PUBLIC SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)

if(NBL_BUILD_IMGUI)
	set(NBL_IMGUI_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
	set(NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT "${THIRD_PARTY_SOURCE_DIR}/imgui_test_engine")
	set(NBL_IMGUI_TEST_ENGINE_ROOT "${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}/imgui_test_engine")
	set(NBL_IMGUI_TEST_SUITE_ROOT "${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}/imgui_test_suite")
	set(NBL_IMPLOT_ROOT "${NBL_IMGUI_TEST_SUITE_ROOT}/thirdparty/implot")

	add_library(imgui STATIC
		"${NBL_IMGUI_ROOT}/imconfig.h"
		"${NBL_IMGUI_ROOT}/imgui_demo.cpp"
		"${NBL_IMGUI_ROOT}/imgui_draw.cpp"
		"${NBL_IMGUI_ROOT}/imgui_internal.h"
		"${NBL_IMGUI_ROOT}/imgui_tables.cpp"
		"${NBL_IMGUI_ROOT}/imgui_widgets.cpp"
		"${NBL_IMGUI_ROOT}/imgui.cpp"
		"${NBL_IMGUI_ROOT}/misc/freetype/imgui_freetype.cpp"
		"${NBL_IMGUI_ROOT}/imgui.h"
		"${NBL_IMGUI_ROOT}/misc/cpp/imgui_stdlib.cpp"
		"${NBL_IMGUI_ROOT}/misc/cpp/imgui_stdlib.h"
		"${NBL_IMGUI_ROOT}/imstb_rectpack.h"
		"${NBL_IMGUI_ROOT}/imstb_textedit.h"
		"${NBL_IMGUI_ROOT}/imstb_truetype.h"
	)
	
	target_link_libraries(imgui PUBLIC freetype)
	
	target_include_directories(imgui PUBLIC 
		"${NBL_IMGUI_ROOT}"
		"${NBL_IMGUI_ROOT}/misc/cpp"
		"${NBL_IMGUI_ROOT}/backends"
		"${NBL_IMGUI_TEST_SUITE_ROOT}"
		$<TARGET_PROPERTY:freetype,INCLUDE_DIRECTORIES>
	)

	add_library(implot STATIC
		"${NBL_IMPLOT_ROOT}/implot.h"
		"${NBL_IMPLOT_ROOT}/implot_internal.h"
		"${NBL_IMPLOT_ROOT}/implot.cpp"
		"${NBL_IMPLOT_ROOT}/implot_items.cpp"
		"${NBL_IMPLOT_ROOT}/implot_demo.cpp"
	)
	
	target_include_directories(implot
		PUBLIC $<TARGET_PROPERTY:imgui,INCLUDE_DIRECTORIES>
		PUBLIC "${NBL_IMPLOT_ROOT}"
	)
	
	target_link_libraries(implot PUBLIC imgui)
	target_compile_definitions(implot PUBLIC IMPLOT_DEBUG IMPLOT_DLL_EXPORT)
	set_target_properties(implot PROPERTIES CXX_STANDARD 20)
	
	if(MSVC)
		target_compile_options(implot PRIVATE /MT /W4 /WX /arch:AVX2 /fp:fast /permissive-) 
	else()
		target_compile_options(implot PRIVATE -Wall -Wextra -pedantic -Werror -mavx2 -Ofast)
	endif()

	file(GLOB_RECURSE NBL_TEST_SUITE_SOURCES CONFIGURE_DEPENDS "${NBL_IMGUI_TEST_SUITE_ROOT}/imgui_test*.cpp")
	file(GLOB_RECURSE NBL_TEST_SUITE_SOURCES CONFIGURE_DEPENDS "${NBL_IMGUI_TEST_SUITE_ROOT}/imgui_test*.cpp")

	add_library(imtestsuite STATIC
		${NBL_TEST_SUITE_SOURCES}
	)
	
	target_include_directories(imtestsuite PUBLIC
		"${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}"
		"${NBL_IMGUI_TEST_ENGINE_ROOT}"
		"${NBL_IMGUI_TEST_SUITE_ROOT}"
		$<TARGET_PROPERTY:imgui,INCLUDE_DIRECTORIES>
		$<TARGET_PROPERTY:implot,INCLUDE_DIRECTORIES>
	)
	
	set_target_properties(imtestsuite PROPERTIES CXX_STANDARD 14) # NOTE! THOSE TESTS DO NOT COMPILE WITH HIGHER STANDARDS SO WE WRAP SOURCES INTO LIBRARY COMPILED WITH LOWER ONE
	target_link_libraries(imtestsuite PUBLIC implot)
	
	file(GLOB_RECURSE NBL_TEST_ENGINE_SOURCES CONFIGURE_DEPENDS "${NBL_IMGUI_TEST_ENGINE_ROOT}/*.cpp")
	
	add_library(imtestengine STATIC
		${NBL_TEST_ENGINE_SOURCES}
	)
	
	target_include_directories(imtestengine PUBLIC
		$<TARGET_PROPERTY:imtestsuite,INCLUDE_DIRECTORIES>
	)
	
	target_link_libraries(imtestengine PUBLIC imtestsuite)
	
	# imguizmo
	add_library(imguizmo 
		"${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/GraphEditor.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImCurveEdit.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImGradient.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImGuizmo.cpp"
		"${CMAKE_CURRENT_SOURCE_DIR}/imguizmo/ImSequencer.cpp"
	)

	target_include_directories(imguizmo PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/imguizmo>)
	target_link_libraries(imguizmo PUBLIC imgui)

	# note we override imgui config with our own
	set(NBL_IMGUI_USER_CONFIG_FILEPATH "${NBL_IMGUI_ROOT}/nabla_imconfig.h")

	if(NOT EXISTS "${NBL_IMGUI_USER_CONFIG_FILEPATH}")
		message(FATAL_ERROR "\"${NBL_IMGUI_USER_CONFIG_FILEPATH}\" doesn't exist!")
	endif()
	
	target_compile_definitions(imgui PUBLIC
		IMGUI_USER_CONFIG="${NBL_IMGUI_USER_CONFIG_FILEPATH}"
	)
endif()

# TODO check for updates + better move to separate file if possible (if CMake provided use its target)
add_library(aesGladman STATIC
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

add_library(phmap INTERFACE) # do not add subdirectory yourself, the author doesn't use CMAKE_CURRENT_SOURCE_DIR for sources hence it will throw errors for someone trying to include our exports
target_include_directories(phmap INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/parallel-hashmap")

add_subdirectory(argparse EXCLUDE_FROM_ALL)

set(GLM_TEST_ENABLE_SIMD_SSE4_2 ON)
set(GLM_TEST_ENABLE OFF)
add_subdirectory(glm EXCLUDE_FROM_ALL)

if (NBL_BUILD_MITSUBA_LOADER)
   set(BUILD_tools OFF)
   set(BUILD_examples OFF)
   set(BUILD_tests OFF)
   set(BUILD_shared OFF)
   set(BUILD_doc OFF)
   set(USE_libbsd OFF)
   set(INSTALL OFF)
   set(MSVC_USE_STATIC_CRT ON)
   add_subdirectory(libexpat/expat EXCLUDE_FROM_ALL)
endif()

add_library(renderdoc INTERFACE)
target_include_directories(renderdoc INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/renderdoc")

# minimum required to build Nabla

set(NBL_IMMUTABLE_TARGETS # note we have a compiler here we do compile and to simply say - we do not want to break things so we do not override their compile options
	dxcConfig
	dxcompiler
	LLVMSupport
	LLVMMSSupport
	SPIRV-Tools-static
	SPIRV-Tools-opt
)

set(NBL_3RDPARTY_TARGETS
	${NBL_IMMUTABLE_TARGETS}
	renderdoc
	lzma
	lz4
	aesGladman
	spirv_cross
	png_static
	pngConfig
	zlibstatic
	zlibConfig
	shaderc_util
	shaderc
	jpeg-static
	jpegConfig
	bzip2
	simdjson
	nlohmann_json
	glslang
	OGLCompiler 
	OSDependent
	MachineIndependent
	GenericCodeGen
	SPIRV
	Imath
	ImathConfig
	freetype
	${NBL_MSDFGEN_TARGETS}
	blake3
	${NBL_BOOST_TARGETS}
	boostConfig
	glm
	volk
	Vulkan-Headers
	argparse # used with x256Hash src/nbl/exe target
	phmap
)

# deps which build is toggled or depends on the platform/toolset used, part of Nabla library

if(TARGET CUDA::toolkit)
	list(APPEND NBL_3RDPARTY_TARGETS CUDA::toolkit)
endif()

if(TARGET DisplaySystemProxy)
	list(APPEND NBL_3RDPARTY_TARGETS DisplaySystemProxy)
endif()

if(UNIX)
	list(APPEND NBL_3RDPARTY_TARGETS atomic)
endif()

if (NBL_BUILD_IMGUI)
	list(APPEND NBL_3RDPARTY_TARGETS imgui implot imtestsuite imtestengine imguizmo)
endif()

if(ANDROID)
	list(APPEND NBL_3RDPARTY_TARGETS android_native_app_glue android log)
endif()

if(_NBL_COMPILE_WITH_OPEN_EXR_)
	list(APPEND NBL_3RDPARTY_TARGETS OpenEXR libdeflate_static OpenEXRUtil OpenEXRCore Iex IlmThread IlmThreadConfig)
endif()

if(_NBL_COMPILE_WITH_GLI_)
	list(APPEND NBL_3RDPARTY_TARGETS gli)
endif()

if(NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_3RDPARTY_TARGETS expat) # TODO: check if its an extension or integrated with Nabla directly
endif()

if(ENABLE_HLSL)
	list(APPEND NBL_3RDPARTY_TARGETS HLSL)
endif()

# extensions which are not part of Nabla library

if(NBL_BUILD_IMGUI)
	list(APPEND NBL_EXTENSIONS_TARGETS imgui implot imguizmo)
endif()

if(NBL_BUILD_MSDF)
	list(APPEND NBL_EXTENSIONS_TARGETS freetype ${NBL_MSDFGEN_TARGETS})
endif()

if(NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_EXTENSIONS_TARGETS expact)
endif()

function(NBL_ADJUST_FOLDER NBL_TARGET NBL_PREFIX KEEP_PRESENT_PREFIX)
	get_target_property(NBL_FOLDER "${NBL_TARGET}" FOLDER)
			
	if(NBL_3RDPARTY_ENABLE_FOLDER_GROUPING)
		if(NBL_FOLDER AND KEEP_PRESENT_PREFIX)
			set_target_properties("${NBL_TARGET}" PROPERTIES FOLDER "${NBL_PREFIX}/${NBL_FOLDER}")
		else()
			set_target_properties("${NBL_TARGET}" PROPERTIES FOLDER "${NBL_PREFIX}")
		endif()
	else()
		set_property(TARGET "${NBL_TARGET}" PROPERTY FOLDER)
	endif()
endfunction()

NBL_GET_ALL_TARGETS(NBL_ALL_3RDPARTY_TARGETS)
foreach(NBL_TARGET IN LISTS NBL_ALL_3RDPARTY_TARGETS)
    get_target_property(NBL_TARGET_SOURCE_DIR ${NBL_TARGET} SOURCE_DIR)
	set(NBL_3RDPARTY_IMMUTABLE_TARGET_SRC_PATTERNS "${NBL_ROOT_PATH}/3rdparty/dxc*")
    NBL_MATCH_PATTERNS("${NBL_TARGET_SOURCE_DIR}" NBL_3RDPARTY_IMMUTABLE_TARGET_SRC_PATTERNS MATCH_FOUND)

    if(MATCH_FOUND)
        list(APPEND NBL_IMMUTABLE_TARGETS ${NBL_TARGET})
    else()
        list(APPEND NBL_MUTABLE_TARGETS ${NBL_TARGET})
    endif()
endforeach()

foreach(NBL_TARGET IN LISTS NBL_ALL_3RDPARTY_TARGETS)
	NBL_ADJUST_FOLDER(${NBL_TARGET} 3rdparty/excluded ON) # first pass to group all targets in IDE
	set_target_properties(${NBL_TARGET} PROPERTIES EXCLUDE_FROM_ALL TRUE) # in case somebody forgots to exclude subdirectory force all as excluded
endforeach()

function(NBL_FINALIZE_TARGETS NBL_TARGETS_LIST_NAME NBL_INTERFACE NBL_EXPORT_INCLUDE_OUTPUT)
	if(NBL_3RDPARTY_EXPORT_TO_BUILD_TREE)
		if(EXISTS "${NBL_EXPORT_INCLUDE_OUTPUT}")
			file(REMOVE "${NBL_EXPORT_INCLUDE_OUTPUT}") # flush
		endif()
	endif()

	foreach(NBL_TARGET IN LISTS ${NBL_TARGETS_LIST_NAME})
		nbl_handle_runtime_lib_properties(${NBL_TARGET})
		set_target_properties(${NBL_TARGET} PROPERTIES EXCLUDE_FROM_ALL FALSE) # pull all targets we listed ourselves to expose them in 3rdparty solution
		
		NBL_ADJUST_FOLDER(${NBL_TARGET} ${NBL_INTERFACE} OFF) # second pass to assign to interface's group
		
		if(MSVC AND NBL_SANITIZE_ADDRESS)
			set_target_properties(${NBL_TARGET} PROPERTIES COMPILE_OPTIONS /fsanitize=address)
		endif()
		
		target_include_directories(${NBL_INTERFACE} 
			INTERFACE $<TARGET_PROPERTY:${NBL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>
			INTERFACE $<TARGET_PROPERTY:${NBL_TARGET},INCLUDE_DIRECTORIES>
		)
		
		get_target_property(NBL_TARGET_TYPE ${NBL_TARGET} TYPE)
		if(NOT NBL_TARGET_TYPE STREQUAL "INTERFACE_LIBRARY")
			get_target_property(IS_IMPORTED ${NBL_TARGET} IMPORTED)
			
			if(NOT IS_IMPORTED AND NOT NBL_TARGET IN_LIST NBL_IMMUTABLE_TARGETS)
				if(NBL_FORCE_RELEASE_3RDPARTY) # priority over RWDI
					nbl_adjust_flags(TARGET ${NBL_TARGET} MAP_RELEASE Release MAP_RELWITHDEBINFO Release MAP_DEBUG Release)
				elseif(NBL_FORCE_RELWITHDEBINFO_3RDPARTY)
					nbl_adjust_flags(TARGET ${NBL_TARGET} MAP_RELEASE RelWithDebInfo MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG RelWithDebInfo)
				else()
					nbl_adjust_flags(TARGET ${NBL_TARGET} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
				endif()
				
				if(MSVC)
					target_compile_definitions(${NBL_TARGET} PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR) # temporary till nbl_adjust_definitions has nbl_adjust_flags's modes
				endif()
			endif()
		endif()
		
		if(NBL_3RDPARTY_EXPORT_TO_BUILD_TREE)
			export(TARGETS ${NBL_TARGET} APPEND FILE "${NBL_EXPORT_INCLUDE_OUTPUT}")
		endif()
	endforeach()
	
	target_link_libraries(${NBL_INTERFACE} INTERFACE "${${NBL_TARGETS_LIST_NAME}}")
endfunction()

nbl_adjust_definitions()

set(NBL_3RDPARTY_EXPORT_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/${NBL_3RDPARTY_EXPORT_MODULE_NAME}.cmake")
set(NBL_EXTENSIONS_EXPORT_INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/${NBL_EXTENSIONS_EXPORT_MODULE_NAME}.cmake")

NBL_FINALIZE_TARGETS(NBL_3RDPARTY_TARGETS 3rdparty "${NBL_3RDPARTY_EXPORT_INCLUDE}")
NBL_FINALIZE_TARGETS(NBL_EXTENSIONS_TARGETS extensions "${NBL_EXTENSIONS_EXPORT_INCLUDE}")

if(NBL_3RDPARTY_EXPORT_TO_BUILD_TREE)
	export(TARGETS 3rdparty APPEND FILE "${NBL_3RDPARTY_EXPORT_INCLUDE}")
	export(TARGETS extensions APPEND FILE "${NBL_EXTENSIONS_EXPORT_INCLUDE}")
endif()

# nbl_install_dir("${CMAKE_CURRENT_SOURCE_DIR}/parallel-hashmap/parallel_hashmap") # TODO: THIS SHOULD NOT BE EXPOSED

set(NBL_MUTABLE_TARGETS ${NBL_MUTABLE_TARGETS} PARENT_SCOPE)
set(NBL_IMMUTABLE_TARGETS ${NBL_IMMUTABLE_TARGETS} PARENT_SCOPE)
set(NBL_ALL_3RDPARTY_TARGETS ${NBL_ALL_3RDPARTY_TARGETS} PARENT_SCOPE)