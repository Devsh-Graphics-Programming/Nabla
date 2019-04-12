# The module defines:
#
# ``NMAKE_EXECUTABLE``
#	Path to 64bit nmake
# ``NMAKE_FOUND``
#	True if the nmake executable was found
#


if (CMAKE_HOST_WIN32)
	get_filename_component(
		MY_COMPILER_DIR
		${CMAKE_CXX_COMPILER} DIRECTORY
	)
	find_program(NMAKE_EXECUTABLE
		nmake.exe
		PATHS "${MY_COMPILER_DIR}/../../Hostx64/x64" "${MY_COMPILER_DIR}"
	)
	string(COMPARE EQUAL NMAKE_EXECUTABLE NMAKE_EXECUTABLE-NOTFOUND NMAKE_FOUND)
		
	unset(MY_COMPILER_DIR)
else()
	set(NMAKE_EXECUTABLE NMAKE_EXECUTABLE-NOTFOUND)
	set(NMAKE_FOUND 0)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nmake
                                  REQUIRED_VARS NMAKE_EXECUTABLE
)