# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(../cmake/common.cmake)

project(Nabla-3rdparty LANGUAGES CXX C)
enable_language(C CXX ASM ASM_NASM)
set(BUILD_SHARED_LIBS OFF)
set(SKIP_INSTALL_ALL ON)
set(BUILD_TESTING OFF)

option(NBL_FORCE_RELEASE_3RDPARTY "Force map 3rdaprty's configuration regardless Nabla configuration to Release" OFF)
option(NBL_FORCE_RELWITHDEBINFO_3RDPARTY "Force map 3rdaprty's configuration regardless Nabla configuration to RelWithDebInfo" OFF)

# zlib
set(ZLIB_BUILD_SHARED OFF)
set(ZLIB_BUILD_STATIC ON)
set(ZLIB_INSTALL OFF)
set(ZLIB_BUILD_TESTING OFF)
set(ZLIB_BUILD_MINIZIP OFF)
add_subdirectory(zlib zlib EXCLUDE_FROM_ALL)
get_target_property(ZLIB_BINARY_DIR zlibstatic "BINARY_DIR")
set_target_properties(zlibstatic PROPERTIES 
	ARCHIVE_OUTPUT_DIRECTORY $<1:${ZLIB_BINARY_DIR}>
	DEBUG_POSTFIX "d"
	RELWITHDEBINFO_POSTFIX "rwdi"
	OUTPUT_NAME zlibstatic
)
add_library(ZLIB::ZLIB ALIAS zlibstatic) # trick projects which want to use it via alias

# nlohmann_json
set(JSON_BuildTests OFF)
add_subdirectory(nlohmann_json nlohmann_json EXCLUDE_FROM_ALL)

# simdjson
add_subdirectory(simdjson simdjson EXCLUDE_FROM_ALL)

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
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE) #forcing boost to be in release
add_subdirectory(boost boost EXCLUDE_FROM_ALL)
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "" FORCE) #restoring config from boost

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

add_subdirectory(shaderc shaderc EXCLUDE_FROM_ALL)

# libjpeg-turbo
set(NBL_JPEG_CMAKE_OPTIONS
	-DCMAKE_MSVC_RUNTIME_LIBRARY:STATIC=MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<BOOL:${NBL_COMPILER_DYNAMIC_RUNTIME}>:DLL>
	-DCMAKE_POLICY_DEFAULT_CMP0057=NEW
	-DCMAKE_POLICY_DEFAULT_CMP0091=NEW
	-DENABLE_SHARED=OFF
	-DENABLE_STATIC=ON
	-DWITH_CRT_DLL=ON
	-DWITH_SIMD=ON
	-DWITH_MEM_SRCDST=ON
	-DWITH_TURBOJPEG=ON
	-DWITH_FUZZ=OFF
	-DWITH_JAVA=OFF
)

if(NOT NBL_IS_MULTI_CONFIG)
	list(APPEND NBL_JPEG_CMAKE_OPTIONS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
endif()

if(NOT CMAKE_GENERATOR MATCHES "Ninja*")
    list(APPEND NBL_JPEG_CMAKE_OPTIONS -Ax64)
endif()

if(CMAKE_GENERATOR_TOOLSET)
	list(APPEND NBL_JPEG_CMAKE_OPTIONS -T "${CMAKE_GENERATOR_TOOLSET}")
endif()

if(CMAKE_TOOLCHAIN_FILE)
	list(APPEND NBL_JPEG_CMAKE_OPTIONS "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}")
endif()

# TODO: might find an alternative library which supports add_subdirectory, untill then we need to switch to a workaround just like
# we do for DXC due to: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/0b742742c873025e2a127918d4969238ace7ae5b/CMakeLists.txt#L69
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo" -B "${CMAKE_CURRENT_BINARY_DIR}/libjpeg-turbo" -G "${CMAKE_GENERATOR}" ${NBL_JPEG_CMAKE_OPTIONS}
	RESULT_VARIABLE JPEG_CMAKE_RESULT
	OUTPUT_VARIABLE JPEG_CMAKE_STREAM_PIPE
)

if(NOT "${JPEG_CMAKE_RESULT}" STREQUAL "0")
	message(FATAL_ERROR "${JPEG_CMAKE_STREAM_PIPE}")
endif()

set(JPEG_LIB_OUTPUT "${THIRD_PARTY_BINARY_DIR}/libjpeg-turbo/$<CONFIG>/jpeg-static${CMAKE_STATIC_LIBRARY_SUFFIX}")
add_custom_command(OUTPUT ${JPEG_LIB_OUTPUT}
	COMMAND "${CMAKE_COMMAND}" --build "${THIRD_PARTY_BINARY_DIR}/libjpeg-turbo" -t jpeg-static --config $<CONFIG>
	COMMENT "Building libjpeg-turbo static library"
	VERBATIM USES_TERMINAL
)
add_library(jpeg-static INTERFACE ${JPEG_LIB_OUTPUT})
target_include_directories(jpeg-static INTERFACE 
	"${THIRD_PARTY_SOURCE_DIR}/libjpeg-turbo/src"
	"${THIRD_PARTY_BINARY_DIR}/libjpeg-turbo"
)
target_link_libraries(jpeg-static INTERFACE "${JPEG_LIB_OUTPUT}")

# blake3
add_subdirectory(blake/c EXCLUDE_FROM_ALL)

# manipulating libpng cmake options from within cmake script
set(PNG_BUILD_ZLIB OFF)
set(PNG_SHARED OFF)
set(PNG_TESTS OFF)
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


# gli
option(_NBL_COMPILE_WITH_GLI_ "Build with GLI library" ON)
if(_NBL_COMPILE_WITH_GLI_)
	set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
	set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
	set(_OLD_BUILD_TESTING ${BUILD_TESTING})
	set(BUILD_SHARED_LIBS OFF)
	set(BUILD_STATIC_LIBS OFF)
	set(BUILD_TESTING OFF)
	set(GLI_GLM_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/glm")
	add_subdirectory(gli gli EXCLUDE_FROM_ALL)
	set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
	set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
	set(BUILD_TESTING ${_OLD_BUILD_TESTING})
endif()

set(ENABLE_STATIC_LIB ON)
set(ENABLE_SHARED_LIB OFF)
set(ENABLE_EXAMPLES OFF)
set(ENABLE_DOCS OFF)
set(ENABLE_APP OFF)
set(ENABLE_LIB_ONLY ON)
set(ENABLE_TESTS OFF)
set(ENABLE_SUMMARY OFF)
add_subdirectory(bzip2 bzip2 EXCLUDE_FROM_ALL)

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
nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/nabla_git_info.json")
nbl_install_file("${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/dxc_git_info.json")

NBL_GENERATE_GIT_TRACKING_META()

# NGFX
include(ngfx/ngfx.cmake)

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
      expat zlibstatic
   )
   set(MITSUBA_LOADER_DEPENDENTS ${MITSUBA_LOADER_DEPENDENTS} PARENT_SCOPE)
   set(MITSUBA_LOADER_DEPENDENT_LIBS
      expat zlibstatic
   )
   set(MITSUBA_LOADER_DEPENDENT_LIBS ${MITSUBA_LOADER_DEPENDENT_LIBS} PARENT_SCOPE)
endif()

if (NBL_BUILD_BULLET)
	set(BULLET_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/bullet3/src
	)

	set(BULLET_INCLUDE_PATH ${BULLET_INCLUDE_PATH} PARENT_SCOPE)
endif()

add_library(Vulkan-Headers INTERFACE)
target_include_directories(Vulkan-Headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/Vulkan-Headers/include")
add_subdirectory(Vulkan-Tools/vulkaninfo vulkaninfo EXCLUDE_FROM_ALL)

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
				bz2_static
				simdjson
				nlohmann_json
				glslang
				OSDependent
				MachineIndependent
				GenericCodeGen
				SPIRV 
				SPIRV-Tools-static # SPIRV-Tools-shared in case of SHARED lib
				SPIRV-Tools-opt
				vulkaninfo
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

foreach(trgt IN LISTS NBL_3RDPARTY_TARGETS)
		set_property(TARGET ${trgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<BOOL:${NBL_COMPILER_DYNAMIC_RUNTIME}>:DLL>")
		
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

NBL_ADJUST_FOLDERS(3rdaprty)

nbl_install_dir(Vulkan-Headers/include/vulkan)
nbl_install_dir(Vulkan-Headers/include/vk_video)

nbl_install_file_spec(volk/volk.h volk)

nbl_install_dir(gtl/include/gtl)
nbl_install_dir(glm/glm)

nbl_install_file_spec(${CMAKE_CURRENT_BINARY_DIR}/imath/config/ImathConfig.h imath)
nbl_install_dir(imath/src/Imath)

nbl_install_file(blake/c/blake3.h)

nbl_install_file_spec(nlohmann_json/include/nlohmann/json_fwd.hpp nlohmann)
nbl_install_file_spec(nlohmann_json/include/nlohmann/detail/abi_macros.hpp nlohmann/detail)

nbl_install_dir(boost/superproject/libs/preprocessor/include/boost)

nbl_install_file_spec(renderdoc/renderdoc_app.h renderdoc)

nbl_install_file(${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/git_info.h)

# parent scope exports, must be at the end of the file
set(_NBL_3RDPARTY_TARGETS_ 
	${NBL_3RDPARTY_TARGETS}
PARENT_SCOPE)

set(NBL_MSDFGEN_TARGETS
	${NBL_MSDFGEN_TARGETS}
PARENT_SCOPE)
