# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(../cmake/common.cmake)

project(Nabla-3rdparty LANGUAGES CXX C)
enable_language(C CXX ASM ASM_NASM)
set(BUILD_SHARED_LIBS OFF)
set(SKIP_INSTALL_ALL ON)
set(BUILD_TESTING OFF)

option(NBL_FORCE_RELEASE_3RDPARTY "Force map 3rdparty's configuration regardless Nabla configuration to Release" OFF)
option(NBL_FORCE_RELWITHDEBINFO_3RDPARTY "Force map 3rdparty's configuration regardless Nabla configuration to RelWithDebInfo" OFF)

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

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_subdirectory(pstl pstl EXCLUDE_FROM_ALL)
endif()

add_library(lzma OBJECT
	lzma/C/Alloc.c
	lzma/C/LzFind.c
	lzma/C/LzmaDec.c
	lzma/C/LzmaEnc.c
)
target_compile_definitions(lzma PRIVATE _7ZIP_ST)


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
	
	# freetype
	find_package(freetype CONFIG REQUIRED)
	target_link_libraries(imgui PUBLIC Freetype::Freetype)
	
	target_include_directories(imgui PUBLIC 
		"${NBL_IMGUI_ROOT}"
		"${NBL_IMGUI_ROOT}/misc/cpp"
		"${NBL_IMGUI_ROOT}/backends"
		"${NBL_IMGUI_TEST_SUITE_ROOT}"
		# $<TARGET_PROPERTY:freetype,INCLUDE_DIRECTORIES>
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

if (NBL_BUILD_BULLET)
	set(BULLET_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/bullet3/src
	)

	set(BULLET_INCLUDE_PATH ${BULLET_INCLUDE_PATH} PARENT_SCOPE)
endif()

add_subdirectory(Vulkan-Tools/vulkaninfo vulkaninfo EXCLUDE_FROM_ALL)

# Final gather
set(NBL_3RDPARTY_TARGETS
				lzma
				aesGladman
				vulkaninfo
				${NBL_MSDFGEN_TARGETS}
)

#TODO insert more targets that are conditionally built
if (NBL_BUILD_IMGUI)
	list(APPEND NBL_3RDPARTY_TARGETS imgui implot imtestsuite imtestengine imguizmo)
endif()

foreach(trgt IN LISTS NBL_3RDPARTY_TARGETS)
		set_property(TARGET ${trgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<BOOL:${NBL_COMPILER_DYNAMIC_RUNTIME}>:DLL>")
		
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

NBL_ADJUST_FOLDERS(3rdparty)

nbl_install_file_spec(renderdoc/renderdoc_app.h renderdoc)

nbl_install_file(${CMAKE_CURRENT_BINARY_DIR}/git-version-tracking/git_info.h)

# parent scope exports, must be at the end of the file
set(_NBL_3RDPARTY_TARGETS_ 
	${NBL_3RDPARTY_TARGETS}
PARENT_SCOPE)

set(NBL_MSDFGEN_TARGETS
	${NBL_MSDFGEN_TARGETS}
PARENT_SCOPE)
