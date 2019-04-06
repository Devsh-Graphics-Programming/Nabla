find_package(Perl REQUIRED)

set(NMAKE_64_EXECUTABLE NOT_FOUND CACHE FILEPATH "Path to x64 nmake.exe")

set(VS140_SCRIPTS_DIR $ENV{VS140COMNTOOLS})

add_custom_target(
  openssl_build ALL
  #COMMAND "${VS140_SCRIPTS_DIR}vsvars32.bat"
  COMMAND "${PERL_EXECUTABLE} ./Configure VC-WIN32 no-asm --prefix=${CMAKE_CURRENT_BINARY_DIR}/build"
  COMMAND "${NMAKE_64_EXECUTABLE} -f ms/ntdll.mak"
  COMMAND "${NMAKE_64_EXECUTABLE} -f ms/ntdll.mak install"
  WORKING_DIRECTORY ./openssl
  COMMENT "Building OpenSSL..."
  VERBATIM
)