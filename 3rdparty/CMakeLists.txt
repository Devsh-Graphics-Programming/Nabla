add_subdirectory(SPIRV-Headers)
add_subdirectory(SPIRV-Tools)
add_subdirectory(glslang)
add_subdirectory(shaderc)

#manipulating libjpeg cmake options from within cmake script
OPTION(BUILD_STATIC OFF)
set(BUILD_STATIC ON)
OPTION(BUILD_EXECUTABLES ON)
set(BUILD_EXECUTABLES OFF)
OPTION(BUILD_TESTS ON)
set(BUILD_TESTS OFF)
add_subdirectory(libjpeg)

#manipulating libpng cmake options from within cmake script
option(PNG_BUILD_ZLIB "Custom zlib Location, else find_package is used" OFF)
set(PNG_BUILD_ZLIB ON) #pnglib depends on zlib and zlib is part of Irrlicht
set(ZLIB_INCLUDE_DIR ${THIRD_PARTY_SOURCE_DIR}/zlib) #for pnglib
option(PNG_SHARED "Build shared lib" ON)
set(PNG_SHARED OFF)
option(PNG_TESTS  "Build libpng tests" ON)
set(PNG_TESTS OFF)
add_subdirectory(libpng)

#add_library(libpng OBJECT
#	libpng/png.c
#	libpng/pngerror.c
#	libpng/pngget.c
#	libpng/pngmem.c
#	libpng/pngpread.c
#	libpng/pngread.c
#	libpng/pngrio.c
#	libpng/pngrtran.c
#	libpng/pngrutil.c
#	libpng/pngset.c
#	libpng/pngtrans.c
#	libpng/pngwio.c
#	libpng/pngwrite.c
#	libpng/pngwtran.c
#	libpng/pngwutil.c
#)
#target_include_directories(libpng PUBLIC zlib)

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