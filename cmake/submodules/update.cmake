include(ProcessorCount)
find_package(Git REQUIRED)

option(NBL_UPDATE_GIT_SUBMODULE "Turn this ON to let CMake update all public submodules for you" ON)
option(NBL_FORCE_ON_UPDATE_GIT_SUBMODULE "Submodules will be updated with --force flag if NBL_FORCE_UPDATE_GIT_SUBMODULE is turned ON, use with caution - if there are any uncommited files in submodules' working tree they will be removed!" OFF)
option(NBL_SYNC_ON_UPDATE_GIT_SUBMODULE "Sync initialized submodule paths if NBL_FORCE_UPDATE_GIT_SUBMODULE is turned ON, this is useful when any submodule remote path got modified and you want to apply this modification to your local repository. Turning NBL_FORCE_ON_UPDATE_GIT_SUBMODULE implies this option" OFF)
option(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE "Turn this ON to attempt to update private Nabla submodules" OFF)
option(NBL_UPDATE_GIT_SUBMODULE_NO_SEPARATE_SHELL "Turn this ON to prevent CMake from executing git submodules update or sync in a separate shell - be aware that the interaction with shell will be impossible in case of paraphrase prompt request of your key!" ON)
option(NBL_CI_GIT_SUBMODULES_SHALLOW "" OFF)

if(NOT DEFINED NBL_ROOT_PATH)
	get_filename_component(NBL_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/../../" ABSOLUTE)
endif()

if(NOT DEFINED THIRD_PARTY_SOURCE_DIR)
	set(THIRD_PARTY_SOURCE_DIR "${NBL_ROOT_PATH}/3rdparty")
endif()

if(NOT DEFINED NBL_ROOT_PATH_BINARY)
	set(NBL_ROOT_PATH_BINARY "${NBL_ROOT_PATH}/build/.submodules")
endif()

if(NOT DEFINED NBL_BUILD_EXAMPLES)
	set(NBL_BUILD_EXAMPLES ON)
endif()

function(NBL_UPDATE_SUBMODULES)
	ProcessorCount(_GIT_SUBMODULES_JOBS_AMOUNT_)
	
	if(NBL_CI_GIT_SUBMODULES_SHALLOW)
		set(NBL_SHALLOW "--depth=1")
	else()
		set(NBL_SHALLOW "")
	endif()
	
	if(NBL_FORCE_ON_UPDATE_GIT_SUBMODULE)
		set(NBL_FORCE "--force")
	else()
		set(NBL_FORCE "")
	endif()

	macro(NBL_WRAPPER_COMMAND_EXCLUSIVE GIT_RELATIVE_ENTRY GIT_SUBMODULE_PATH SHOULD_RECURSIVE EXCLUDE_SUBMODULE_PATHS)
		set(EXCLUDE_SUBMODULE_PATHS ${EXCLUDE_SUBMODULE_PATHS})
		set(SHOULD_RECURSIVE ${SHOULD_RECURSIVE})
		
		if("${EXCLUDE_SUBMODULE_PATHS}" STREQUAL "")
			set(NBL_EXCLUDE "")
		else()
			foreach(EXCLUDE_SUBMODULE_PATH ${EXCLUDE_SUBMODULE_PATHS})
				string(APPEND NBL_EXCLUDE "-c submodule.\"${EXCLUDE_SUBMODULE_PATH}\".update=none ")
			endforeach()
			
			string(STRIP "${NBL_EXCLUDE}" NBL_EXCLUDE)
		endif()

		if(SHOULD_RECURSIVE)
			set(_NBL_EXECUTE_COMMAND_ "\"${GIT_EXECUTABLE}\" -C \"${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}\" ${NBL_EXCLUDE} submodule update --init -j ${_GIT_SUBMODULES_JOBS_AMOUNT_} ${NBL_FORCE} --recursive ${NBL_SHALLOW} ${GIT_SUBMODULE_PATH}")
		else()
			set(_NBL_EXECUTE_COMMAND_ "\"${GIT_EXECUTABLE}\" -C \"${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}\" ${NBL_EXCLUDE} submodule update --init -j ${_GIT_SUBMODULES_JOBS_AMOUNT_} ${NBL_FORCE} ${NBL_SHALLOW} ${GIT_SUBMODULE_PATH}")
		endif()
		
		string(APPEND _NBL_UPDATE_SUBMODULES_COMMANDS_ "${_NBL_EXECUTE_COMMAND_}\n")
		
		unset(NBL_EXCLUDE)
	endmacro()
	
	set(_NBL_UPDATE_SUBMODULES_CMD_NAME_ "nbl-update-submodules")
	set(_NBL_UPDATE_SUBMODULES_CMD_FILE_ "${NBL_ROOT_PATH_BINARY}/${_NBL_UPDATE_SUBMODULES_CMD_NAME_}.cmd")
	get_filename_component(_NBL_UPDATE_IMPL_CMAKE_FILE_ "${NBL_ROOT_PATH_BINARY}/${_NBL_UPDATE_SUBMODULES_CMD_NAME_}.cmake" ABSOLUTE)
	
	# Proxy script for inclusive submodule updating
	string(APPEND NBL_IMPL_SCRIPT "set(NBL_ROOT_PATH \"${NBL_ROOT_PATH}\")\nset(_GIT_SUBMODULES_JOBS_AMOUNT_ ${_GIT_SUBMODULES_JOBS_AMOUNT_})\nset(GIT_EXECUTABLE \"${GIT_EXECUTABLE}\")\nset(NBL_SHALLOW \"${NBL_SHALLOW}\")\nset(NBL_FORCE \"${NBL_FORCE}\")\n\n")
	string(APPEND NBL_IMPL_SCRIPT
[=[
if(NOT DEFINED GIT_RELATIVE_ENTRY)
	message(FATAL_ERROR "GIT_RELATIVE_ENTRY must be defined to use this script!")
endif()

if(NOT DEFINED INCLUDE_SUBMODULE_PATHS)
	message(FATAL_ERROR "INCLUDE_SUBMODULE_PATHS must be defined to use this script!")
endif()

# update an inclusive submodule first
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${NBL_ROOT_PATH}" submodule update --init "${GIT_RELATIVE_ENTRY}")

if("${INCLUDE_SUBMODULE_PATHS}" STREQUAL "")
	set(NBL_SUBMODULE_UPDATE_CONFIG_ENTRY "")
else()
	execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}" config --file .gitmodules --get-regexp path
		OUTPUT_VARIABLE NBL_OUTPUT_VARIABLE
	)

	string(REGEX REPLACE "\n" ";" NBL_SUBMODULE_CONFIG_LIST "${NBL_OUTPUT_VARIABLE}")
	
	foreach(NBL_SUBMODULE_NAME ${NBL_SUBMODULE_CONFIG_LIST})
		string(REGEX MATCH "submodule\\.(.*)\\.path" NBL_SUBMODULE_NAME "${NBL_SUBMODULE_NAME}")
		list(APPEND NBL_ALL_SUBMODULES "${CMAKE_MATCH_1}")
	endforeach()
	
	foreach(NBL_SUBMODULE_NAME ${NBL_ALL_SUBMODULES})		
		list(FIND INCLUDE_SUBMODULE_PATHS "${NBL_SUBMODULE_NAME}" NBL_FOUND)
		
		if("${NBL_FOUND}" STREQUAL "-1")
			list(APPEND NBL_CONFIG_SETUP_CMD "-c;submodule.${NBL_SUBMODULE_NAME}.update=none") # filter submodules - only those on the INCLUDE_SUBMODULE_PATHS list will be updated when recursive update is requested, all left will be skipped
		endif()
	endforeach()
endif()

execute_process(COMMAND "${GIT_EXECUTABLE}" ${NBL_CONFIG_SETUP_CMD} submodule update --init -j ${_GIT_SUBMODULES_JOBS_AMOUNT_} --recursive ${NBL_SHALLOW} ${NBL_FORCE}
	WORKING_DIRECTORY "${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}"
)
]=]
)
	file(WRITE "${_NBL_UPDATE_IMPL_CMAKE_FILE_}" "${NBL_IMPL_SCRIPT}")
	
	macro(NBL_WRAPPER_COMMAND_INCLUSIVE GIT_RELATIVE_ENTRY INCLUDE_SUBMODULE_PATHS)
		string(APPEND _NBL_UPDATE_SUBMODULES_COMMANDS_ "\"${CMAKE_COMMAND}\" \"-DGIT_RELATIVE_ENTRY=${GIT_RELATIVE_ENTRY}\" \"-DINCLUDE_SUBMODULE_PATHS=${INCLUDE_SUBMODULE_PATHS}\" -P \"${_NBL_UPDATE_IMPL_CMAKE_FILE_}\"\n")
	endmacro()
	
	if(NBL_UPDATE_GIT_SUBMODULE)
		execute_process(COMMAND ${CMAKE_COMMAND} -E echo "All submodules are about to get updated and initialized in repository because NBL_UPDATE_GIT_SUBMODULE is turned ON!")
		
		include("${THIRD_PARTY_SOURCE_DIR}/boost/dep/wave.cmake")
		
		macro(NBL_IMPL_INIT_COMMON_SUBMODULES)
			# 3rdparty except boost & gltf
			set(NBL_3RDPARTY_MODULES_TO_SKIP
				3rdparty/boost/superproject # a lot of submodules we don't use
				3rdparty/glTFSampleModels # more then 2GB waste of space (disk + .gitmodules data)
			)
			NBL_WRAPPER_COMMAND_EXCLUSIVE("" ./3rdparty TRUE "${NBL_3RDPARTY_MODULES_TO_SKIP}")
			
			# boost's 3rdparties, special case
			set(NBL_BOOST_LIBS_TO_INIT ${NBL_BOOST_LIBS} wave numeric_conversion) # wave and all of its deps, numeric_conversion is nested in conversion submodule (for some reason boostdep tool doesn't output it properly)
			foreach(NBL_TARGET ${NBL_BOOST_LIBS_TO_INIT})
				list(APPEND NBL_BOOST_SUBMODULES_TO_INIT ${NBL_TARGET})
			endforeach()
			NBL_WRAPPER_COMMAND_INCLUSIVE(3rdparty/boost/superproject "${NBL_BOOST_SUBMODULES_TO_INIT}")
			
			# tests
			NBL_WRAPPER_COMMAND_EXCLUSIVE("" ./tests FALSE "")
		endmacro()
		
		NBL_IMPL_INIT_COMMON_SUBMODULES()
		
		if(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE)
			NBL_WRAPPER_COMMAND_EXCLUSIVE("" ./examples_tests TRUE "")
		else()
			# NBL_WRAPPER_COMMAND_EXCLUSIVE("" ./ci TRUE "") TODO: enable it once we merge Ditt, etc
			
			# examples and their media
			if(NBL_BUILD_EXAMPLES)
				NBL_WRAPPER_COMMAND_EXCLUSIVE("" ./examples_tests FALSE "")
				NBL_WRAPPER_COMMAND_EXCLUSIVE(examples_tests ./media FALSE "")
			endif()
		endif()
				
		file(WRITE "${_NBL_UPDATE_SUBMODULES_CMD_FILE_}" "${_NBL_UPDATE_SUBMODULES_COMMANDS_}")

		if(WIN32)
			if(NBL_UPDATE_GIT_SUBMODULE_NO_SEPARATE_SHELL)
				set(UPDATE_COMMAND
					nbl-update-submodules.cmd
				)
			
				execute_process(COMMAND ${UPDATE_COMMAND}
					WORKING_DIRECTORY "${NBL_ROOT_PATH_BINARY}"
					RESULT_VARIABLE _NBL_TMP_RET_CODE_
				)
			else()
				find_package(GitBash REQUIRED)
		
				execute_process(COMMAND "${GIT_BASH_EXECUTABLE}" "-c"
[=[
>&2 echo ""
clear
./nbl-update-submodules.cmd 2>&1 | tee nbl-update-submodules.log
sleep 1
clear
tput setaf 2; echo -e "Submodules have been updated! 
Created nbl-update-submodules.log in your build directory."
]=]
					WORKING_DIRECTORY ${NBL_ROOT_PATH_BINARY}
					OUTPUT_VARIABLE _NBL_TMP_OUTPUT_
					RESULT_VARIABLE _NBL_TMP_RET_CODE_
					OUTPUT_STRIP_TRAILING_WHITESPACE
					ERROR_STRIP_TRAILING_WHITESPACE
				)
				
				unset(_NBL_TMP_OUTPUT_)
				unset(_NBL_TMP_RET_CODE_)
			
				message(STATUS "Generated \"${NBL_ROOT_PATH_BINARY}/nbl-update-submodules.log\"")
			endif()
			
			message(STATUS "Submodules have been updated!")
		else()
			execute_process(COMMAND "${_NBL_UPDATE_SUBMODULES_CMD_FILE_}")
		endif()
	else()
		execute_process(COMMAND ${CMAKE_COMMAND} -E echo "NBL_UPDATE_GIT_SUBMODULE is turned OFF therefore submodules won't get updated.")
	endif()
endfunction()

NBL_UPDATE_SUBMODULES()