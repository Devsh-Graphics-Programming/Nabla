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

# freetype2
set(FT_WITH_HARFBUZZ OFF)
add_subdirectory(freetype2 freetype2 EXCLUDE_FROM_ALL)
add_library(Freetype::Freetype ALIAS freetype) # alias target for MSDFGEN not trying to find freetype system-wide but rather to use the logical target available in our build tree

# msdfgen
set(MSDFGEN_USE_VCPKG OFF)
set(MSDFGEN_USE_SKIA OFF)
set(MSDFGEN_CORE_ONLY OFF) # if turned on then MSDFGEN will be built without dependencies like freetype or skia
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
NBL_TARGET_FORCE_ASSEMBLER_EXECUTABLE(simd ASM_NASM libjpeg-turbo/simd/)

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

# Imath
set(IMATH_INSTALL OFF)
add_subdirectory(imath EXCLUDE_FROM_ALL)

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

# cmake-git-version-tracking
# note that checking if a repository is dirty may cost build time (especially a lot if like us you have a lot of submodules) - by default we run with all checks but if you want to increase build time iterations I recommend to exclude this check
option(GIT_EXCLUDE_IS_DIRTY "Exclude IS_DIRTY from git tracking checks, will increase build time iterations at the expense of the meta information loss" OFF)
add_subdirectory(git-version-tracking EXCLUDE_FROM_ALL)
NBL_ADD_GIT_TRACKING_META_LIBRARY(nabla "${NBL_ROOT_PATH}")
NBL_ADD_GIT_TRACKING_META_LIBRARY(dxc "${CMAKE_CURRENT_SOURCE_DIR}/dxc/dxc")
NBL_GENERATE_GIT_TRACKING_META()

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
				Imath
				freetype
				${NBL_MSDFGEN_TARGETS}
				blake3
				${NBL_BOOST_TARGETS}
)
if (_NBL_COMPILE_WITH_OPEN_EXR_)
	list(APPEND NBL_3RDPARTY_TARGETS
		OpenEXR libdeflate_static OpenEXRUtil OpenEXRCore Iex IlmThread
	)
endif()

#TODO insert more targets that are conditionally built
if (NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_3RDPARTY_TARGETS expat)
endif()
if (NBL_BUILD_IMGUI)
	list(APPEND NBL_3RDPARTY_TARGETS imgui implot imtestsuite imtestengine imguizmo)
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
			
			if(MSVC)
				target_compile_definitions(${trgt} PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR) # temporary till nbl_adjust_definitions has nbl_adjust_flags's modes
			endif()
		endif()
endforeach()
nbl_adjust_definitions()

add_custom_target(3rdparty)
add_dependencies(3rdparty ${NBL_3RDPARTY_TARGETS})

nbl_install_dir("${CMAKE_CURRENT_SOURCE_DIR}/parallel-hashmap/parallel_hashmap")

# parent scope exports, must be at the end of the file
set(_NBL_3RDPARTY_TARGETS_ 
	${NBL_3RDPARTY_TARGETS}
PARENT_SCOPE)

set(NBL_MSDFGEN_TARGETS
	${NBL_MSDFGEN_TARGETS}
PARENT_SCOPE)
