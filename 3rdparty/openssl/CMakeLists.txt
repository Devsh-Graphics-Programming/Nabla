find_package(Git)

if(GIT_FOUND)
	# run those sequentially
	execute_process(COMMAND "${GIT_EXECUTABLE} fetch"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
	)
	execute_process(COMMAND "${GIT_EXECUTABLE} clean -fd"
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
	)
	execute_process(COMMAND "${GIT_EXECUTABLE} reset --hard origin/OpenSSL_1_0_2-stable"
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
	)
else()
	message(WARNING "Git not found! Might be you have to update git-submodules manually")
endif()
# maybe you (Crisspl) can patch the OpenSSL source for the preprocessor MSVC bug here?


if(WIN32)
	find_package(Perl REQUIRED)
	
	get_filename_component(
		MY_COMPILER_DIR
		${CMAKE_CXX_COMPILER} DIRECTORY
	)
   	find_file(
		VCVARS64_BAT vcvars64.bat
		"${MY_COMPILER_DIR}/../../../../../../Auxiliary/Build"
	)
	if ("${VCVARS64_BAT}" STREQUAL "VCVARS64_BAT-NOTFOUND")
		message(FATAL_ERROR "Path to vcvars64.bat must be provided manually")
	endif()
   	
	find_package(nmake REQUIRED)

	add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib"
		COMMAND "${VCVARS64_BAT}"
		COMMAND "${PERL_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/openssl/Configure" VC-WIN64A no-asm --prefix=${CMAKE_CURRENT_BINARY_DIR}/build
		COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/do_win64a.bat"
		COMMAND "${NMAKE_EXECUTABLE}" -f "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/nt.mak"
		COMMAND "${NMAKE_EXECUTABLE}" -f "${CMAKE_CURRENT_SOURCE_DIR}/openssl/ms/nt.mak" install
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
		COMMENT "Building OpenSSL..."
		VERBATIM
	)
	add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib")
else()
	add_custom_command(
		OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a"
		COMMAND ./config no-shared --prefix=${CMAKE_CURRENT_BINARY_DIR}/build --openssldir=${CMAKE_CURRENT_BINARY_DIR}/build
		COMMAND make -j64 # for the threadripper
		COMMAND make install
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
		COMMENT "Building OpenSSL..."
		VERBATIM
	)
	add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a")
endif()
