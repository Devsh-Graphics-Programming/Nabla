# Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
# This file is part of the "Nabla Engine".
# For conditions of distribution and use, see copyright notice in nabla.h

include(../cmake/common.cmake)

# mandatory
option(NBL_UPDATE_GIT_SUBMODULE "" ON)

if(NBL_UPDATE_GIT_SUBMODULE)
	update_git_submodule(./SDL2)
	update_git_submodule(./volk)
	update_git_submodule(./SPIRV-Headers)
	update_git_submodule(./SPIRV-Tools)
	update_git_submodule(./glslang)
	update_git_submodule(./shaderc)
	update_git_submodule(./bzip2)
	update_git_submodule(./lz4)
	update_git_submodule(./spirv_cross)
	update_git_submodule(./zlib)
	update_git_submodule(./openexr)
	update_git_submodule(./utfcpp)
	update_git_submodule(./convertutf)
	update_git_submodule(./libpng)
	update_git_submodule(./openssl)
	update_git_submodule(./libjpeg-turbo)
	update_git_submodule(./libjpeg) # soon to be killed
	update_git_submodule(./parallel-hashmap)
	update_git_submodule(./gli)
	# exceptions that get automatically cloned
	update_git_submodule(./glm)
	update_git_submodule(./freetype2)
	update_git_submodule(./portable-file-dialogs)
endif()

# openssl
add_subdirectory(openssl openssl EXCLUDE_FROM_ALL)

# volk
if(DEFINED ENV{VULKAN_SDK})
	# workaround to make sure vulkan library will not be linked in examples
	set(VOLK_PULL_IN_VULKAN OFF CACHE INTERNAL "" FORCE)
	
	if (WIN32)
		set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
	endif()
	add_subdirectory(volk volk EXCLUDE_FROM_ALL)
	
    target_include_directories(volk PUBLIC "$ENV{VULKAN_SDK}/include")
    target_include_directories(volk_headers INTERFACE "$ENV{VULKAN_SDK}/include")
endif()

# SDL2
if (NBL_COMPILE_WITH_SDL2)
	set(SDL_STATIC_ENABLED_BY_DEFAULT ON)
	set(SDL_SHARED_ENABLED_BY_DEFAULT OFF)
	set(SDL_THREADS_ENABLED_BY_DEFAULT OFF)
	set(SDL_LOADSO_ENABLED_BY_DEFAULT ON)
	# should I enable this
	set(SDL_DLOPEN_ENABLED_BY_DEFAULT OFF)
	set(SDL_SUBSYSTEMS "Audio Video Events Joystick Haptic Power Loadso CPUinfo Filesystem Sensor")
	set(SDL_STATIC_PIC ON)
	set(SDL_ATOMIC OFF)
	set(SDL_FILESYSTEM OFF)
	set(SDL_TIMERS OFF)
	# oh why doesn't SDL2 prefix these options?
	set(MMX OFF)
	set(3DNOW OFF)
	set(VIDEO_DIRECTFB OFF)
	set(VIDEO_OPENGL ON)
	set(PTHREADS OFF)
	set(RENDER_D3D OFF)
	set(RENDER_METAL OFF)
	set(VIDEO_VIVANTE OFF)
	# Metal?
	set(VIDEO_KMSDRM OFF)
	set(VIDEO_OFFSCREEN ON)
	set(VIDEO_X11_XCURSOR ON)
	set(VIDEO_X11_XINERAMA OFF)
	set(VIDEO_X11_XINPUT ON)
	set(VIDEO_X11_XRANDR ON)
	set(VIDEO_X11_XSCRNSAVER ON)
	set(VIDEO_X11_XSHAPE ON)
	set(VIDEO_X11_XVM OFF)
	# make sure as much gets shared as possible
	set(OSS OFF)
	set(ALSA_SHARED ON)
	set(JACK_SHARED ON)
	set(ESD_SHARED ON)
	set(PULSEAUDIO_SHARED ON)
	set(ARTS_SHARED ON)
	set(NAS_SHARED ON)
	set(SNDIO_SHARED ON)
	set(FUSIONSOUND_SHARED ON)
	set(LIBSAMPLERATE_SHARED ON)
	set(WAYLAND_SHARED ON)
	set(X11_SHARED ON)
	add_subdirectory(SDL2 SDL2 EXCLUDE_FROM_ALL)
endif()

find_package(OpenGL OPTIONAL_COMPONENTS EGL)
find_package(Wayland)
#[[if (OpenGL_EGL_FOUND)
	message(FATAL_ERROR "FOUND EGL!!")
else()
	message(FATAL_ERROR "COULDNT FIND EGL!!")
endif()]]
# EGL
set(EGL_NO_GLEW ON CACHE INTERNAL "" FORCE)
#if (EGL_NO_GLEW)
#	message(FATAL_ERROR "EGL_NO_GLEW")
#else()
#	message(FATAL_ERROR "NO EGL_NO_GLEW")
#endif()
if (OpenGL_EGL_FOUND AND UNIX AND NOT APPLE)
	set(EGL_PASSTHROUGH ON FORCE)
	if (WAYLAND_CLIENT_FOUND)
		set(EGL_USE_WAYLAND ON FORCE)
		message(FATAL_ERROR "Using Wayland backend!!!")
	endif()
endif()
add_subdirectory(EGL/EGL)

# zlib (target is zlibstatic)
set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(_OLD_SKIP_INSTALL_ALL ${SKIP_INSTALL_ALL})
set(BUILD_SHARED_LIBS OFF)
set(SKIP_INSTALL_ALL ON)
add_subdirectory(zlib zlib EXCLUDE_FROM_ALL)
set(SKIP_INSTALL_ALL ${_OLD_SKIP_INSTALL_ALL})
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})

get_target_property(ZLIB_BINARY_DIR zlibstatic "BINARY_DIR")
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
add_subdirectory(SPIRV-Headers SPIRV-Headers EXCLUDE_FROM_ALL)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE INTERNAL "Skip SPIRV-Tools install")
set(SPIRV_SKIP_TESTS ON CACHE INTERNAL "Skip SPIRV-Tools tests")
set(SPIRV_SKIP_EXECUTABLES ON CACHE INTERNAL "Skip SPIRV-Tools executable build")
add_subdirectory(SPIRV-Tools SPIRV-Tools EXCLUDE_FROM_ALL)
set(SKIP_GLSLANG_INSTALL ON CACHE INTERNAL "Skip glslang install")
set(ENABLE_SPVREMAPPER OFF CACHE INTERNAL "SPVRemapper is a spirv compression experimental library")
set(ENABLE_GLSLANG_BINARIES OFF CACHE INTERNAL "Build glslangValidator and spirv-remap?")
set(BUILD_TESTING OFF CACHE INTERNAL "Enable glslang build testing?")
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

#manipulating libjpeg cmake options from within cmake script
#option(ENABLE_SHARED "jpeg-turbo" OFF)
#set(CMAKE_INSTALL_DOCDIR "${CMAKE_CURRENT_BINARY_DIR}/doc" CACHE PATH "For libjpeg-turbo" FORCE)
option(BUILD_STATIC "jpeg" ON)
option(BUILD_EXECUTABLES "jpeg" OFF)
option(BUILD_TESTS "jpeg" OFF)
add_subdirectory(libjpeg libjpeg EXCLUDE_FROM_ALL)

#manipulating libpng cmake options from within cmake script
set(PNG_BUILD_ZLIB ON CACHE INTERNAL "" FORCE) # get ZLIB from our /3rdparty
option(PNG_SHARED "Build shared lib" OFF)
option(PNG_TESTS  "Build libpng tests" OFF)
add_subdirectory(libpng libpng EXCLUDE_FROM_ALL)
add_dependencies(png_static zlibstatic)

#openexr
option(_NBL_COMPILE_WITH_OPEN_EXR_ "Build with OpenEXR library" ON)
if (_NBL_COMPILE_WITH_OPEN_EXR_)
	set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
	set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
	set(_OLD_BUILD_TESTING ${BUILD_TESTING})
	set(BUILD_SHARED_LIBS OFF)
	set(BUILD_STATIC_LIBS OFF)
	set(BUILD_TESTING OFF)
	set(PYILMBASE_ENABLE OFF CACHE STRING "" FORCE)
	set(OPENEXR_BUILD_UTILS OFF CACHE STRING "" FORCE)
	set(OPENEXR_VIEWERS_ENABLE OFF CACHE STRING "" FORCE)
	add_subdirectory(openexr openexr EXCLUDE_FROM_ALL)
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

nbl_adjust_flags()

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
	spirv_cross/spirv_cfg.cpp
	spirv_cross/spirv_cross.cpp
	spirv_cross/spirv_glsl.cpp
	spirv_cross/spirv_reflect.cpp
	spirv_cross/spirv_parser.cpp
	spirv_cross/spirv_cross_parsed_ir.cpp
)
target_compile_definitions(spirv_cross PUBLIC SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)

add_library(convert_utf OBJECT
	convertutf/ConvertUTF.cpp
)

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

if (NBL_BUILD_CEGUI) # I absolutely hate and despise GLM
	set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
	set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
	set(BUILD_SHARED_LIBS OFF)
	set(BUILD_STATIC_LIBS OFF)
	option(GLM_TEST_ENABLE_CXX_14 "Enable C++ 14" ON)
	if (NBL_FAST_MATH)
	   option(GLM_TEST_ENABLE_FAST_MATH "Enable fast math optimizations" ON)
	endif()
	option(GLM_TEST_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" ON)
	option(GLM_TEST_ENABLE "Build unit tests" OFF)
	add_subdirectory(glm EXCLUDE_FROM_ALL)
	set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
	set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})
endif()

option(GLM_TEST_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" ON)
option(GLM_TEST_ENABLE "Build unit tests" OFF)
#add_subdirectory(glm EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})

if (NBL_BUILD_MITSUBA_LOADER)
	if(NBL_UPDATE_GIT_SUBMODULE)
		update_git_submodule(./libexpat)
	endif()

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

if (NBL_BUILD_CEGUI)
	if(NBL_UPDATE_GIT_SUBMODULE)
		update_git_submodule(./libexpat)
		update_git_submodule(./CEGUI/pcre)
		update_git_submodule(./CEGUI/glew)
		update_git_submodule(./CEGUI/cegui)
	endif()
	add_subdirectory(CEGUI)
	set(CEGUI_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include
		${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include
	)
	set(CEGUI_INCLUDE_PATH ${CEGUI_INCLUDE_PATH} PARENT_SCOPE)

	# surely there has to be a cleaner way of doing this
	if (MSVC)
		set(CEGUI_LIBS
			${CEGUI_DEPS_INSTALL_PATH}/lib/libglew32.lib
			${CEGUI_DEPS_INSTALL_PATH}/lib/freetype.lib
			${CEGUI_DEPS_INSTALL_PATH}/lib/expat.lib
			${CEGUI_DEPS_INSTALL_PATH}/lib/pcre.lib
			${CEGUI_DEPS_INSTALL_PATH}/lib/pcrecpp.lib
			${CEGUI_DEPS_INSTALL_PATH}/lib/pcreposix.lib
			Winmm.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIOpenGLRenderer-0_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIBase-0_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIExpatParser_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUISTBImageCodec_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICoreWindowRendererSet_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICommonDialogs-0_Static.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIOpenGLRenderer-0_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIBase-0_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIExpatParser_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUISTBImageCodec_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICoreWindowRendererSet_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICommonDialogs-0_Static_d.lib
			debug Dbghelp.lib
		)
	else()
		set(CEGUI_LIBS
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUIOpenGLRenderer-0_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUIBase-0_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUIExpatParser_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUISTBImageCodec_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUICoreWindowRendererSet_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUICommonDialogs-0_Static.a
			${CEGUI_DEPS_INSTALL_PATH}/lib/libGLEW.a
			${CEGUI_DEPS_INSTALL_PATH}/lib/libfreetype.a
			${CEGUI_DEPS_INSTALL_PATH}/lib/libexpat.a
			${CEGUI_DEPS_INSTALL_PATH}/lib/libpcre.a
			${CEGUI_DEPS_INSTALL_PATH}/lib/libpcrecpp.a
			${CEGUI_DEPS_INSTALL_PATH}/lib/libpcreposix.a
		)
	endif()
	set(CEGUI_LIBS ${CEGUI_LIBS} PARENT_SCOPE)

	#[[ unsure if needed for now
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/include/
			DESTINATION ./include/nbl/ext/CEGUI/3rdparty
			CONFIGURATIONS Release
		)
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/include/
			DESTINATION ./debug/include/nbl/ext/CEGUI/3rdparty
			CONFIGURATIONS Debug
		)
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/lib/
			DESTINATION ./lib/nbl/ext/CEGUI
			CONFIGURATIONS Release
		)
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/lib/
			DESTINATION ./debug/lib/nbl/ext/CEGUI
			CONFIGURATIONS Debug
		)
	]]#

	# CEGUI headers
	install(
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./include/nbl/ext/CEGUI
		CONFIGURATIONS Release
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./debug/include/nbl/ext/CEGUI
		CONFIGURATIONS Debug
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./relwithdebinfo/include/nbl/ext/CEGUI
		CONFIGURATIONS RelWithDebInfo
	)
	# CEGUI generated headers
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./include/nbl/ext/CEGUI
		CONFIGURATIONS Release
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./debug/include/nbl/ext/CEGUI
		CONFIGURATIONS Debug
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./relwithdebinfo/include/nbl/ext/CEGUI
		CONFIGURATIONS RelWithDebInfo
	)
	# CEGUI libs
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/
		DESTINATION ./lib/nbl/ext/CEGUI
		CONFIGURATIONS Release
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/
		DESTINATION ./debug/lib/nbl/ext/CEGUI
		CONFIGURATIONS Debug
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/
		DESTINATION ./relwithdebinfo/lib/nbl/ext/CEGUI
		CONFIGURATIONS RelWithDebInfo
	)
endif()

if (NBL_BUILD_RADEON_RAYS)
	if(NBL_UPDATE_GIT_SUBMODULE)
		update_git_submodule(./radeonrays)
	endif()

	option(RR_NO_TESTS "RADEON_RAYS: don't do tests" ON)
	option(RR_ENABLE_STATIC "RADEON_RAYS: compile into a static library" ON)
	option(RR_EMBED_KERNELS "RADEON_RAYS: so we don't have to go looking for OpenCL kernel code" ON)
	option(RR_USE_VULKAN "RADEON_RAYS: use Vulkan (only if you have RR 3.0)" OFF)
	set(MSVC_USE_STATIC_CRT ON CACHE BOOL "Use /MT flag (static CRT) when compiling in MSVC")
	add_subdirectory(radeonrays EXCLUDE_FROM_ALL)

	set(RADEON_RAYS_DEPENDENTS
		RadeonRays
		CLW
		Calc
	)
	set(RADEON_RAYS_DEPENDENTS ${RADEON_RAYS_DEPENDENTS} PARENT_SCOPE)
	set(RADEON_RAYS_DEPENDENT_LIBS
		RadeonRays
	)
	set(RADEON_RAYS_DEPENDENT_LIBS ${RADEON_RAYS_DEPENDENT_LIBS} PARENT_SCOPE)
endif()

if (NBL_BUILD_BULLET)
	if(NBL_UPDATE_GIT_SUBMODULE)
		update_git_submodule(./bullet3)
	endif()
	set(BULLET_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/bullet3/src
	)

	set(BULLET_INCLUDE_PATH ${BULLET_INCLUDE_PATH} PARENT_SCOPE)
endif()


if (NBL_COMPILE_WITH_CUDA)
	if(NBL_UPDATE_GIT_SUBMODULE)
		update_git_submodule(./jitify)
	endif()
endif()

set(NBL_3RDPARTY_TARGETS
				lzma
				lz4
				aesGladman
				OpenSSL
				spirv_cross
				png_static
				zlibstatic
				shaderc_util
				shaderc
				jpeg
				convert_utf
				bzip2			
				glslang
				OGLCompiler 
				OSDependent
				MachineIndependent
				GenericCodeGen
				SPIRV 
				SPIRV-Tools
				SPIRV-Tools-opt
				egl 
)
if (_NBL_COMPILE_WITH_OPEN_EXR_)
	list(APPEND NBL_3RDPARTY_TARGETS
	    IlmBaseConfig
		IlmImfConfig
		Imath
		IlmThread
		IlmImf
		Half
		Iex
		IexMath
	)
endif()
if (NBL_COMPILE_WITH_SDL2 )
	list(APPEND NBL_3RDPARTY_TARGETS
	    SDL2-static
	)
endif()
#TODO insert more targets that are conditionally built
if (NBL_BUILD_MITSUBA_LOADER)
	list(APPEND NBL_3RDPARTY_TARGETS expat)
endif()
if (NBL_BUILD_RADEON_RAYS)
	list(APPEND NBL_3RDPARTY_TARGETS ${RADEON_RAYS_DEPENDENTS})
endif()
if(ENABLE_HLSL)
	list(APPEND NBL_3RDPARTY_TARGETS HLSL)
endif()

foreach(trgt IN LISTS NBL_3RDPARTY_TARGETS)
		set_property(TARGET ${trgt} PROPERTY
             MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endforeach()

set(_NBL_3RDPARTY_TARGETS_ ${NBL_3RDPARTY_TARGETS} PARENT_SCOPE)

# parallel hashmap headers
install(
	DIRECTORY ./parallel-hashmap/parallel_hashmap 
	DESTINATION ./include/parallel-hashmap
	CONFIGURATIONS Release
)
install(
	DIRECTORY ./parallel-hashmap//parallel_hashmap 
	DESTINATION ./debug/include/parallel-hashmap
	CONFIGURATIONS Debug
)
install(
	DIRECTORY ./parallel-hashmap/parallel_hashmap 
	DESTINATION ./relwithdebinfo/include/parallel-hashmap
	CONFIGURATIONS RelWithDebInfo
)