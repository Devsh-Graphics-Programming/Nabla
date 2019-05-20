execute_process(COMMAND git submodule update --init --recursive
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


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
add_subdirectory(glslang)

set(SHADERC_SKIP_TESTS ON CACHE BOOL "Skip shaderc tests?")
set(SHADERC_SKIP_INSTALL ON CACHE BOOL "Install shaderc?")

# if it doesn't work without the `touch` on Linux, then fetch the latest submodule head of shaderc and try again
# https://github.com/google/shaderc/issues/568
if (UNIX)
	file(WRITE ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/libshaderc_combined.a "")
endif()
add_subdirectory(shaderc)

add_subdirectory(openssl)

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

if (IRR_BUILD_CEGUI)
	add_subdirectory(CEGUI)
	set(CEGUI_INCLUDE_PATH
		${CMAKE_CURRENT_SOURCE_DIR}/CEGUI/cegui/cegui/include
		${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/cegui/include
	)
	set(CEGUI_INCLUDE_PATH ${CEGUI_INCLUDE_PATH} PARENT_SCOPE)

	if (MSVC)
		set(CEGUI_LIBS
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIOpenGLRenderer-0_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIBase-0_Static.lib
			optimized ${CEGUI_DEPS_INSTALL_PATH}/lib/glew.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIExpatParser_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUISTBImageCodec_Static.lib
			optimized ${CEGUI_DEPS_INSTALL_PATH}/lib/freetype.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICoreWindowRendererSet_Static.lib
			optimized ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICommonDialogs-0_Static.lib
			optimized ${CEGUI_DEPS_INSTALL_PATH}/lib/libexpat.lib
			optimized ${CEGUI_DEPS_INSTALL_PATH}/lib/pcre.lib
			optimized Winmm.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIOpenGLRenderer-0_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIBase-0_Static_d.lib
			debug ${CEGUI_DEPS_INSTALL_PATH}/lib/glew_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUIExpatParser_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUISTBImageCodec_Static_d.lib
			debug ${CEGUI_DEPS_INSTALL_PATH}/lib/freetype_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICoreWindowRendererSet_Static_d.lib
			debug ${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/CEGUICommonDialogs-0_Static_d.lib
			debug ${CEGUI_DEPS_INSTALL_PATH}/lib/libexpat_d.lib
			debug ${CEGUI_DEPS_INSTALL_PATH}/lib/pcre_d.lib
			debug Dbghelp.lib
			debug Winmm.lib
			PARENT_SCOPE
		)
	else()
    find_package(EXPAT REQUIRED)
    find_package(Freetype REQUIRED)
    find_package(GLEW REQUIRED)
    find_package(PCRE REQUIRED)
		set(CEGUI_LIBS
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUIOpenGLRenderer-0_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUIBase-0_Static.a
			${GLEW_LIBRARIES}
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUIExpatParser_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUISTBImageCodec_Static.a
			${FREETYPE_LIBRARIES}
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUICoreWindowRendererSet_Static.a
			${CMAKE_CURRENT_BINARY_DIR}/CEGUI/cegui/lib/libCEGUICommonDialogs-0_Static.a
			${EXPAT_LIBRARIES}
			${PCRE_LIBRARIES}
		)
		set(CEGUI_LIBS ${CEGUI_LIBS} PARENT_SCOPE)
	endif()

	# Only MSVC builds and installs CEGUI dependencies
	if (MSVC)
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
	endif()

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
