if(NOT DEFINED GIT_EXECUTABLE)
	message(FATAL_ERROR "GIT_EXECUTABLE must be defined!")
endif()

cmake_path(GET GIT_EXECUTABLE PARENT_PATH _GIT_CMD_INSTALL_DIR_) # /cmd directory
cmake_path(GET _GIT_CMD_INSTALL_DIR_ PARENT_PATH _GIT_INSTALL_DIR_) # /Git directory

find_program(GIT_BASH_EXECUTABLE NAMES "git-bash.exe" PATHS "${_GIT_INSTALL_DIR_}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GitBash REQUIRED_VARS GIT_BASH_EXECUTABLE)
