# Copyright (c) 2019 DevSH Graphics Programming Sp. z O.O.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(../cmake/common.cmake)


# mandatory
update_git_submodule(./SPIRV-Headers)
update_git_submodule(./SPIRV-Tools)
update_git_submodule(./glslang)
update_git_submodule(./shaderc)
update_git_submodule(./bzip2)
update_git_submodule(./lz4)
update_git_submodule(./spirv_cross)
update_git_submodule(./zlib)
update_git_submodule(./utfcpp)
update_git_submodule(./convertutf)
update_git_submodule(./libpng)
update_git_submodule(./openssl)
update_git_submodule(./libjpeg-turbo)
update_git_submodule(./libjpeg) # soon to be killed
# exceptions that get automatically cloned
update_git_submodule(./glm)
update_git_submodule(./freetype2)
update_git_submodule(./portable-file-dialogs)



set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "Skip SPIRV-Headers examples")
add_subdirectory(SPIRV-Headers SPIRV-Headers EXCLUDE_FROM_ALL)
set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "Skip SPIRV-Tools install")
set(SPIRV_SKIP_TESTS ON CACHE BOOL "Skip SPIRV-Tools tests")
set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "Skip SPIRV-Tools executable build")
add_subdirectory(SPIRV-Tools SPIRV-Tools EXCLUDE_FROM_ALL)
set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "Skip glslang install")
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "SPVRemapper is a spirv compression experimental library")
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "Build glslangValidator and spirv-remap?")
set(BUILD_TESTING OFF CACHE BOOL "Enable glslang build testing?")
add_subdirectory(glslang glslang EXCLUDE_FROM_ALL)

set(SHADERC_SKIP_TESTS ON CACHE BOOL "Skip shaderc tests?")
set(SHADERC_SKIP_INSTALL ON CACHE BOOL "Install shaderc?")

# if it doesn't work without the `touch` on Linux, then fetch the latest submodule head of shaderc and try again
# https://github.com/google/shaderc/issues/568
if (UNIX)
	file(WRITE ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/libshaderc_combined.a "")
endif()
add_subdirectory(shaderc shaderc EXCLUDE_FROM_ALL)

add_subdirectory(openssl openssl EXCLUDE_FROM_ALL)

#manipulating libjpeg cmake options from within cmake script
#option(ENABLE_SHARED "jpeg-turbo" OFF)
#set(CMAKE_INSTALL_DOCDIR "${CMAKE_CURRENT_BINARY_DIR}/doc" CACHE PATH "For libjpeg-turbo" FORCE)
option(BUILD_STATIC "jpeg" ON)
option(BUILD_EXECUTABLES "jpeg" OFF)
option(BUILD_TESTS "jpeg" OFF)
add_subdirectory(libjpeg libjpeg EXCLUDE_FROM_ALL)

#manipulating libpng cmake options from within cmake script
option(PNG_BUILD_ZLIB "Custom zlib Location, else find_package is used" OFF)
set(PNG_BUILD_ZLIB ON)
set(ZLIB_INCLUDE_DIR ${THIRD_PARTY_SOURCE_DIR}/zlib) #for pnglib
set(ZLIB_LIBRARY "$<TARGET_OBJECTS:zlib>") #for pnglib
option(PNG_SHARED "Build shared lib" OFF)
option(PNG_TESTS  "Build libpng tests" OFF)
add_subdirectory(libpng libpng EXCLUDE_FROM_ALL)
add_dependencies(png_static zlib) #CMake doesn't get it itself

irr_adjust_flags()

add_library(zlib OBJECT
	zlib/adler32.c
	zlib/crc32.c
	zlib/deflate.c
	zlib/infback.c
	zlib/inffast.c
	zlib/inflate.c
	zlib/inftrees.c
	zlib/trees.c
	zlib/uncompr.c
	#zlib/zcompress.c
	zlib/zutil.c
)
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
target_compile_definitions(spirv_cross PRIVATE SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)

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

set(_OLD_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
set(_OLD_BUILD_STATIC_LIBS ${BUILD_STATIC_LIBS})
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS OFF)
option(GLM_TEST_ENABLE_CXX_14 "Enable C++ 14" ON)
if (IRR_FAST_MATH)
   option(GLM_TEST_ENABLE_FAST_MATH "Enable fast math optimizations" ON)
endif()
option(GLM_TEST_ENABLE_SIMD_SSE4_2 "Enable SSE 4.2 optimizations" ON)
option(GLM_TEST_ENABLE "Build unit tests" OFF)
add_subdirectory(glm EXCLUDE_FROM_ALL)
set(BUILD_SHARED_LIBS ${_OLD_BUILD_SHARED_LIBS})
set(BUILD_STATIC_LIBS ${_OLD_BUILD_STATIC_LIBS})

if (IRR_BUILD_MITSUBA_LOADER)
   update_git_submodule(./libexpat)

   option(BUILD_tools "EXPAT: build the xmlwf tool for expat library" OFF)
   option(BUILD_examples "EXPAT: build the examples for expat library" OFF)
   option(BUILD_tests "EXPAT: build the tests for expat library" OFF)
   option(BUILD_shared "EXPAT: build a shared expat library" OFF)
   option(BUILD_doc "EXPAT: build man page for xmlwf" OFF)
   option(USE_libbsd "EXPAT: utilize libbsd (for arc4random_buf)" OFF)
   option(INSTALL "EXPAT: install expat files in cmake install target" OFF)
   set(MSVC_USE_STATIC_CRT ON CACHE BOOL "Use /MT flag (static CRT) when compiling in MSVC")
   add_subdirectory(libexpat/expat EXCLUDE_FROM_ALL)
endif()

if (IRR_BUILD_CEGUI)
	update_git_submodule(./libexpat)
	update_git_submodule(./CEGUI/pcre)
	update_git_submodule(./CEGUI/glew)
	update_git_submodule(./CEGUI/cegui)
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
		set(CEGUI_LIBS ${CEGUI_LIBS} PARENT_SCOPE)
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
		set(CEGUI_LIBS ${CEGUI_LIBS} PARENT_SCOPE)
	endif()

	#[[ unsure if needed for now
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/include/
			DESTINATION ./include/irr/ext/CEGUI/3rdparty
			CONFIGURATIONS Release
		)
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/include/
			DESTINATION ./debug/include/irr/ext/CEGUI/3rdparty
			CONFIGURATIONS Debug
		)
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/lib/
			DESTINATION ./lib/irr/ext/CEGUI
			CONFIGURATIONS Release
		)
		install(
			DIRECTORY ${CEGUI_DEPS_INSTALL_PATH}/lib/
			DESTINATION ./debug/lib/irr/ext/CEGUI
			CONFIGURATIONS Debug
		)
	]]#

	# CEGUI headers
	install(
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./include/irr/ext/CEGUI
		CONFIGURATIONS Release
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./debug/include/irr/ext/CEGUI
		CONFIGURATIONS Debug
	)
	# CEGUI generated headers
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./include/irr/ext/CEGUI
		CONFIGURATIONS Release
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include/CEGUI
		DESTINATION ./debug/include/irr/ext/CEGUI
		CONFIGURATIONS Debug
	)
	# CEGUI libs
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/
		DESTINATION ./lib/irr/ext/CEGUI
		CONFIGURATIONS Release
	)
	install(
		DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/
		DESTINATION ./debug/lib/irr/ext/CEGUI
		CONFIGURATIONS Debug
	)
endif()

if (IRR_BUILD_BULLET)
	execute_process(COMMAND git submodule update --init --recursive ./bullet3
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
	)
	set(BULLET_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/bullet3/src
	)

	set(BULLET_INCLUDE_PATH ${BULLET_INCLUDE_PATH} PARENT_SCOPE)
endif()
