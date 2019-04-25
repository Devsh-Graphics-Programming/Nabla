execute_process(COMMAND git submodule update --init --recursive
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)


option(SPIRV_HEADERS_SKIP_EXAMPLES "SPIRV-Headers examples" ON)
add_subdirectory(SPIRV-Headers)
option(SKIP_SPIRV_TOOLS_INSTALL "SPIRV-Tools install skip" ON)
option(SPIRV_SKIP_TESTS "SPIRV-Tools tests skip" ON)
option(SPIRV_SKIP_EXECUTABLES "SPIRV-Tools executable build skip" ON)
add_subdirectory(SPIRV-Tools)
option(SKIP_GLSLANG_INSTALL "glslang install skip" ON)
option(ENABLE_SPVREMAPPER "spvremapper is a spirv compression experimental library" OFF)
option(ENABLE_GLSLANG_BINARIES "build glslangValidator and spirv-remap?" OFF)
set(BUILD_TESTING OFF CACHE BOOL "glslang build testing")
add_subdirectory(glslang)

OPTION(SHADERC_SKIP_TESTS "skip shaderc tests?" ON)
OPTION(SHADERC_SKIP_INSTALL "install shaderc?" ON)
# if it doesn't work without the `touch` on Linux, then fetch the latest submodule head of shaderc and try again
#if (UNIX)
#	file(WRITE ${THIRD_PARTY_SOURCE_DIR}/shaderc/libshaderc/libshaderc_combined.a "") #https://github.com/google/shaderc/issues/568
#endif()
add_subdirectory(shaderc)

add_subdirectory(openssl)

#manipulating libjpeg cmake options from within cmake script
option(BUILD_STATIC "jpeg" ON)
option(BUILD_EXECUTABLES "jpeg" OFF)
option(BUILD_TESTS "jpeg" OFF)
add_subdirectory(libjpeg)

#manipulating libpng cmake options from within cmake script
option(PNG_BUILD_ZLIB "Custom zlib Location, else find_package is used" ON)
set(ZLIB_INCLUDE_DIR ${THIRD_PARTY_SOURCE_DIR}/zlib) #for pnglib
option(PNG_SHARED "Build shared lib" OFF)
option(PNG_TESTS  "Build libpng tests" OFF)
add_subdirectory(libpng)

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
