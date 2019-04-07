find_package(Perl REQUIRED)

set(NMAKE_64_EXECUTABLE NOT_FOUND CACHE FILEPATH "Path to x64 nmake.exe")

set(VS140_SCRIPTS_DIR $ENV{VS140COMNTOOLS})

add_custom_command(
	OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib"
	COMMAND "${VS140_SCRIPTS_DIR}vsvars32.bat"
	COMMAND "${PERL_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/openssl/Configure" VC-WIN32 no-asm --prefix="${CMAKE_CURRENT_BINARY_DIR}/build"
	COMMAND "${NMAKE_64_EXECUTABLE}" -f "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/ntdll.mak"
	COMMAND "${NMAKE_64_EXECUTABLE}" -f "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/ntdll.mak" install
	#WORKING_DIRECTORY ./openssl
	COMMENT "Building OpenSSL..."
	VERBATIM
)
add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib")