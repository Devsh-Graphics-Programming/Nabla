include(ProcessorCount)
find_package(Git REQUIRED)

option(NBL_UPDATE_GIT_SUBMODULE "Turn this ON to let CMake update all public submodules for you" ON)
option(NBL_FORCE_ON_UPDATE_GIT_SUBMODULE "Submodules will be updated with --force flag if NBL_FORCE_UPDATE_GIT_SUBMODULE is turned ON + ALL SUBMODULES' CONTENT will be WIPED, use with caution - if there are any uncommited files in submodules' working tree they will be removed!" OFF)
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

	function(NBL_EXCLUSIVE_UPDATE_EXECUTE GIT_RELATIVE_ENTRY GIT_SUBMODULE_PATH SHOULD_RECURSIVE EXCLUDE_SUBMODULE_PATHS)
		if(EXCLUDE_SUBMODULE_PATHS STREQUAL "")
			set(NBL_EXCLUDE "")
		else()
			foreach(EXCLUDE_SUBMODULE_PATH ${EXCLUDE_SUBMODULE_PATHS})
				string(APPEND NBL_EXCLUDE "-c;submodule.\"${EXCLUDE_SUBMODULE_PATH}\".update=none;")
			endforeach()
			
			string(STRIP "${NBL_EXCLUDE}" NBL_EXCLUDE)
		endif()

		list(APPEND NBL_COMMAND "${GIT_EXECUTABLE}" -C "${NBL_ROOT_PATH}/${GIT_RELATIVE_ENTRY}" ${NBL_EXCLUDE} submodule update --init -j ${_GIT_SUBMODULES_JOBS_AMOUNT_} ${NBL_FORCE})

		if(SHOULD_RECURSIVE)
			list(APPEND NBL_COMMAND --recursive)
		endif()

		list(APPEND NBL_COMMAND ${NBL_SHALLOW} ${GIT_SUBMODULE_PATH})

		execute_process(COMMAND ${NBL_COMMAND})
	endfunction()
	
	function(NBL_INCLUSIVE_UPDATE_EXECUTE GIT_RELATIVE_ENTRY INCLUDE_SUBMODULE_PATHS)
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
	endfunction()
	
	if(NBL_UPDATE_GIT_SUBMODULE)
		execute_process(COMMAND ${CMAKE_COMMAND} -E echo "NBL_UPDATE_GIT_SUBMODULE is turned ON, updating submodules.")
		
		if(NBL_SYNC_ON_UPDATE_GIT_SUBMODULE)
			execute_process(COMMAND "${GIT_EXECUTABLE}" submodule sync --recursive
				WORKING_DIRECTORY "${NBL_ROOT_PATH}"
			)
		endif()

		if(NBL_FORCE_ON_UPDATE_GIT_SUBMODULE)
			execute_process(COMMAND "${GIT_EXECUTABLE}" submodule foreach --recursive "${GIT_EXECUTABLE}" clean -fdx
				WORKING_DIRECTORY "${NBL_ROOT_PATH}"
			)

			execute_process(COMMAND "${GIT_EXECUTABLE}" submodule foreach --recursive "${GIT_EXECUTABLE}" reset --hard
				WORKING_DIRECTORY "${NBL_ROOT_PATH}"
			)
		endif()

		include("${THIRD_PARTY_SOURCE_DIR}/boost/dep/wave.cmake")
		
		# 3rdparty except boost & gltf
		set(NBL_3RDPARTY_MODULES_TO_SKIP
			3rdparty/boost/superproject # a lot of submodules we don't use
			3rdparty/glTFSampleModels # more then 2GB waste of space (disk + .gitmodules data)
		)

		NBL_EXCLUSIVE_UPDATE_EXECUTE("" ./3rdparty TRUE "${NBL_3RDPARTY_MODULES_TO_SKIP}")
		
		# boost's 3rdparties, special case
		set(NBL_BOOST_LIBS_TO_INIT ${NBL_BOOST_LIBS} wave numeric_conversion) # wave and all of its deps, numeric_conversion is nested in conversion submodule (for some reason boostdep tool doesn't output it properly)
		foreach(NBL_TARGET ${NBL_BOOST_LIBS_TO_INIT})
			list(APPEND NBL_BOOST_SUBMODULES_TO_INIT ${NBL_TARGET})
		endforeach()

		NBL_INCLUSIVE_UPDATE_EXECUTE(3rdparty/boost/superproject "${NBL_BOOST_SUBMODULES_TO_INIT}")
		
		# tests
		NBL_EXCLUSIVE_UPDATE_EXECUTE("" ./tests FALSE "")
		
		if(NBL_UPDATE_GIT_SUBMODULE_INCLUDE_PRIVATE)
			NBL_EXCLUSIVE_UPDATE_EXECUTE("" ./examples_tests TRUE "")
		else()
			# NBL_EXCLUSIVE_UPDATE_EXECUTE("" ./ci TRUE "") TODO: enable it once we merge Ditt, etc
			
			# examples and their media
			if(NBL_BUILD_EXAMPLES)
				NBL_EXCLUSIVE_UPDATE_EXECUTE("" ./examples_tests FALSE "")
				NBL_EXCLUSIVE_UPDATE_EXECUTE(examples_tests ./media FALSE "")
			endif()
		endif()
	else()
		execute_process(COMMAND "${CMAKE_COMMAND}" -E echo "NBL_UPDATE_GIT_SUBMODULE is turned OFF, submodules won't get updated.")
	endif()
endfunction()

NBL_UPDATE_SUBMODULES()