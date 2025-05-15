find_package(Git REQUIRED)

option(NBL_UPDATE_GIT_SUBMODULE "Turn ON to update submodules, only public by default" ON)
option(NBL_FORCE_ON_UPDATE_GIT_SUBMODULE "NBL_UPDATE_GIT_SUBMODULE logic with --force flag" OFF)
option(NBL_SYNC_ON_UPDATE_GIT_SUBMODULE "Sync submodule URLs" OFF)
option(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE "NBL_UPDATE_GIT_SUBMODULE logic but includes private submodules, for Nabla devs" OFF)
option(NBL_SUBMODULES_SHALLOW "NBL_UPDATE_GIT_SUBMODULE logic with --depth=1" OFF)

if(NBL_UPDATE_GIT_SUBMODULE)
block()
	get_filename_component(NBL_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
	set(THIRD_PARTY_SOURCE_DIR "${NBL_ROOT_PATH}/3rdparty")

	if(NOT DEFINED NBL_ROOT_PATH_BINARY)
		set(NBL_ROOT_PATH_BINARY "${NBL_ROOT_PATH}/build/.submodules")
	endif()

	if(NOT DEFINED NBL_BUILD_EXAMPLES)
		set(NBL_BUILD_EXAMPLES ON)
	endif()

	# we force HTTPS traffic for all *public* submodules we update from CMake
	# NOTE: it *doesn't* rewrite destination URLs after checkout, if you eg. 
	# clone with SSH you end up with it anyway, this way your private key 
	# is never involved during CMake configuration, unless you
	# use NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE

	# Private refs (*), exclude from public update
	list(APPEND NBL_CONFIG_SUBMODULE -c submodule.\"Ditt-Reference-Scenes\".update=none)

	unset(NBL_UPDATE_OPTIONS)

	if(NBL_SUBMODULES_SHALLOW)
		list(APPEND NBL_UPDATE_OPTIONS --depth=1)
	endif()

	if(NBL_FORCE_ON_UPDATE_GIT_SUBMODULE)
		list(APPEND NBL_UPDATE_OPTIONS --force)
	endif()

	if(NOT NBL_BUILD_EXAMPLES)
		list(APPEND NBL_CONFIG_SUBMODULE -c submodule.\"examples_tests\".update=none)
	endif()

	macro(NBL_GIT_COMMAND)
		execute_process(COMMAND "${GIT_EXECUTABLE}" ${ARGV})
	endmacro()

	if(NBL_SYNC_ON_UPDATE_GIT_SUBMODULE)
		message(STATUS "Syncing Public submodules")
		NBL_GIT_COMMAND(${NBL_CONFIG_SUBMODULE} submodule sync --recursive WORKING_DIRECTORY "${NBL_ROOT_PATH}")
	endif()
	
	message(STATUS "Updating Public submodules")
	NBL_GIT_COMMAND(-c url.https://github.com/.insteadOf=git@github.com: ${NBL_CONFIG_SUBMODULE} submodule update --init --recursive ${NBL_UPDATE_OPTIONS} WORKING_DIRECTORY "${NBL_ROOT_PATH}")

	if(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE)
		# NOTE: your git must be installed with default Git Bash as shell 
		# otherwise it *may* fail, whether it works depends on your agent setup

		find_package(GitBash REQUIRED)

		macro(NBL_GIT_BASH_COMMAND)
			execute_process(COMMAND "${GIT_BASH_EXECUTABLE}" "-c" ${ARGV})
		endmacro()

		message(STATUS "Updating Private submodules")
		string(REPLACE ";" " " NBL_UPDATE_OPTIONS "${NBL_UPDATE_OPTIONS}")
		set(LOG_FILE "${NBL_ROOT_PATH_BINARY}/nbl-update-private-submodules.log")
		set(BASH_CMD
[=[
>&2 echo ""
clear
{	
	echo "=== $(date) :: Starting private submodule update ==="
	git -c submodule.Ditt-Reference-Scenes.update=checkout -C @NBL_ROOT_PATH@/examples_tests/media submodule update --init Ditt-Reference-Scenes @NBL_UPDATE_OPTIONS@
	# more private submodule here

	echo "=== $(date) :: Created @LOG_FILE@ in your build directory. ==="
	echo "=== $(date) :: Finished private submodule update ==="
} 2>&1 | tee @LOG_FILE@
clear
]=]
		)
		string(CONFIGURE "${BASH_CMD}" BASH_CMD)
		NBL_GIT_BASH_COMMAND("${BASH_CMD}" OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE RES)
		file(READ "${LOG_FILE}" LOG_CONTENT)
		message(STATUS "${LOG_CONTENT}")
	endif()
endblock()
endif()