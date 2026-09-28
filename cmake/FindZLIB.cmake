if(NOT TARGET ZLIB::ZLIB)
	message(FATAL_ERROR "define ZLIB::ZLIB target!")
endif()

set(ZLIB_FOUND TRUE)
set(ZLIB_LIBRARY ZLIB::ZLIB)
set(ZLIB_INCLUDE_DIR "${THIRD_PARTY_SOURCE_DIR}/zlib;${THIRD_PARTY_BINARY_DIR}/zlib")

# Upstream FindZLIB also sets the plural forms; libpng's genout.cmake.in reads
# @ZLIB_INCLUDE_DIRS@ and fails to preprocess pnglibconf.c without it.
set(ZLIB_INCLUDE_DIRS "${ZLIB_INCLUDE_DIR}")
set(ZLIB_LIBRARIES "${ZLIB_LIBRARY}")