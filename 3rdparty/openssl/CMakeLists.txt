if(WIN32)
	find_package(Perl REQUIRED)

	set(NMAKE_64_EXECUTABLE NOT_FOUND CACHE FILEPATH "Path to x64 nmake.exe")
	set(VCVARS64_BATCH_FILE NOT_FOUND CACHE FILEPATH 
		"Path to vcvar64.bat; usually something like C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/vcvars64.bat"
	)

	if (NOT ${NMAKE_64_EXECUTABLE})
		message(FATAL_ERROR "Path to nmake must be provided manually")
	endif()
	if (NOT ${VCVARS64_BATCH_FILE})
		message(FATAL_ERROR "Path to vcvars64.bat must be provided manually")
	endif()


	add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib"
		COMMAND "${VCVARS64_BATCH_FILE}" 
		COMMAND "${PERL_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/openssl/Configure" VC-WIN64A no-asm --prefix=${CMAKE_CURRENT_BINARY_DIR}/build
		COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/do_win64a.bat"
		COMMAND "${NMAKE_64_EXECUTABLE}" -f "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/nt.mak"
		COMMAND "${NMAKE_64_EXECUTABLE}" -f "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/nt.mak" install
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
		COMMENT "Building OpenSSL..."
		VERBATIM
	)
	add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib")
else()
	add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a"
		./config no-shared --prefix=${CMAKE_CURRENT_BINARY_DIR}/build --openssldir=${CMAKE_CURRENT_BINARY_DIR}/build --api=1.0.2
		COMMAND make -j64 # for the threadripper
		COMMAND make install
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
		COMMENT "Building OpenSSL..."
		VERBATIM
	)
	add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a")
endif()
