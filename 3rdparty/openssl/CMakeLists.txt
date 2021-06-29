find_package(Git)

if(GIT_FOUND)
	# run those sequentially
	execute_process(COMMAND "${GIT_EXECUTABLE} fetch"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
	)
	execute_process(COMMAND "${GIT_EXECUTABLE} clean -fdx"
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
	)
	execute_process(COMMAND "${GIT_EXECUTABLE} reset --hard origin/OpenSSL_1_1_1-stable"
		WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
	)
else()
	message(WARNING "Git not found! Might be you have to update git-submodules manually")
endif()

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
	
	add_library(OpenSSL INTERFACE)
	
	target_link_libraries(OpenSSL INTERFACE
		"${CMAKE_CURRENT_BINARY_DIR}/build/lib/libeay32.lib" 
		"${CMAKE_CURRENT_BINARY_DIR}/build/lib/ssleay32.lib"
	)
	
elseif(ANDROID) # cross-compile
	if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Linux")
		find_package(Perl REQUIRED)
		
		set(ENV{PATH} "${ANDROID_NDK_ROOT_PATH}/toolchains/llvm/prebuilt/linux-x86_64/bin:${ANDROID_NDK_ROOT_PATH}/toolchains/x86_64-4.9/prebuilt/linux-86_64/bin:${ANDROID_NDK_ROOT_PATH}")
	
		# ATTENTION - it used to work but ENV variable doesn't work while building https://cmake.org/cmake/help/latest/command/set.html#set-environment-variable, TODO check after changes one more time
		add_custom_command(
			OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a"
			COMMAND make clean
			# not sure if linux-generic64 is right (https://stackoverflow.com/questions/26779531/openssl-for-android-x86-64-and-mips64)
			# but android-x86_64 is not even recognized (should be according to this https://github.com/openssl/openssl/blob/master/NOTES-ANDROID.md)
			COMMAND "${PERL_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/openssl/Configure" linux-generic64 no-asm no-shared -D__ANDROID_API__=${ANDROID_API_LEVEL} --openssldir=${CMAKE_CURRENT_BINARY_DIR}/build
			COMMAND make -j64
			COMMAND make install
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/openssl"
			VERBATIM
		)
		
		add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a")
	
		add_library(OpenSSL INTERFACE)
	
		target_link_libraries(OpenSSL INTERFACE
			"${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" 
			"${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a"
		)
	elseif("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Windows")
		#find_package(Perl REQUIRED) probably when bellow TODO is fixed we will have to use strawberry perl anyways due to default lack of Windows Console x32 module in Active Perl
		
		set(WRAPPER_CODE	
			"cd ${CMAKE_CURRENT_SOURCE_DIR}/openssl\n"
			"set ANDROID_NDK_HOME=${ANDROID_NDK_ROOT_PATH}\n"
			"set PATH=%ANDROID_NDK_HOME%/toolchains/llvm/prebuilt/windows-x86_64/bin;%ANDROID_NDK_HOME%/toolchains/x86_64-4.9/prebuilt/windows-x86_64/bin;%ANDROID_NDK_HOME%/prebuilt/windows-x86_64/bin;%PATH%\n"
			"make.exe clean\n"
			"${PERL_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/openssl/Configure android-x86_64 no-asm no-shared -D__ANDROID_API__=${ANDROID_API_LEVEL} --openssldir=${CMAKE_CURRENT_BINARY_DIR}/build\n"
			"make.exe -j64\n"
			"make.exe install"
		)
	
		file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/build/openssl_build_wrapper.bat" ${WRAPPER_CODE})
		
		# TODO - when openssl works with clang as windows host
		
		#add_custom_command(
		#	OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a"
		#	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/build/openssl_build_wrapper.bat
		#	VERBATIM
		#)
		
		#add_custom_target(openssl_build DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" "${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a")
	endif()
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
	
	add_library(OpenSSL INTERFACE)
	
	target_link_libraries(OpenSSL INTERFACE
		"${CMAKE_CURRENT_BINARY_DIR}/build/lib/libcrypto.a" 
		"${CMAKE_CURRENT_BINARY_DIR}/build/lib/libssl.a"
	)
endif()
