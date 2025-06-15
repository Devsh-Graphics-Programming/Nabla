get_filename_component(NBL_BOOST_WAVE_DEP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/dep/wave.cmake" ABSOLUTE)

# Boost uses it's own tool for generating dependency list for targets, therefore we 
# can make sure manually added dependency subdirectories for a library are valid
# https://www.boost.org/doc/libs/1_83_0/tools/boostdep/doc/html/index.html#boostdep.introduction.building_boostdep

if(NBL_BOOST_GENERATE_DEP_LIST) # internal, for Nabla devs
	if(NOT WIN32)
		message(FATAL_ERROR "NBL_BOOST_GENERATE_DEP_LIST only for Windows host!")
	endif()

	macro(NBL_BOOST_EXECUTE)
		execute_process(COMMAND ${ARGV} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/superproject")
	endmacro()

	NBL_BOOST_EXECUTE(git config --file .gitmodules --get-regexp path OUTPUT_VARIABLE NBL_OUTPUT_VARIABLE)
	string(REGEX REPLACE "\n" ";" NBL_SUBMODULE_CONFIG_LIST "${NBL_OUTPUT_VARIABLE}")

	foreach(NBL_SUBMODULE_NAME ${NBL_SUBMODULE_CONFIG_LIST})
		string(REGEX MATCH "submodule\\.(.*)\\.path" NBL_SUBMODULE_NAME "${NBL_SUBMODULE_NAME}")
		list(APPEND BOOST_SUBMODULES "${CMAKE_MATCH_1}")
	endforeach()

	# sync & force update of all boost modules first for the tool purpose (sry guys who use the tool, you need to clone all, I want to keep it simple)
	NBL_BOOST_EXECUTE(git submodule sync)
	list(APPEND BOOST_FORCE_ALL_CONFIG -c url.https://github.com/.insteadOf=git@github.com:)
	foreach(SUBMODULE ${BOOST_SUBMODULES})
		list(APPEND BOOST_FORCE_ALL_CONFIG -c submodule.${SUBMODULE}.update=checkout)
	endforeach()

	NBL_BOOST_EXECUTE(git ${BOOST_FORCE_ALL_CONFIG} submodule update --init --recursive -f)

	# build boost dep executable
	set(NBL_BOOSTDEP_EXE "boostdep.exe")
	set(NBL_BOOSTDEP_EXE_FILEPATH "${CMAKE_CURRENT_BINARY_DIR}/superproject/tools/boostdep/bin/${NBL_BOOSTDEP_EXE}")
	if(NOT EXISTS "${NBL_BOOSTDEP_EXE_FILEPATH}")
		NBL_BOOST_EXECUTE(cmd /C bootstrap.bat)
		NBL_BOOST_EXECUTE(cmd /C b2.exe tools/boostdep/build)
		NBL_BOOST_EXECUTE("${CMAKE_COMMAND}" -E copy "./dist/bin/${NBL_BOOSTDEP_EXE}" "${NBL_BOOSTDEP_EXE_FILEPATH}")
		NBL_BOOST_EXECUTE(git clean -fdx)
		NBL_BOOST_EXECUTE(git reset --hard)
	endif()

	# get wave dependency info
	NBL_BOOST_EXECUTE("${NBL_BOOSTDEP_EXE_FILEPATH}" --boost-root "${CMAKE_CURRENT_SOURCE_DIR}/superproject" --brief wave
		OUTPUT_VARIABLE NBL_OUTPUT_VAR
	)

	file(WRITE "${NBL_BOOST_WAVE_DEP_FILE}" "${NBL_OUTPUT_VAR}")

	file(STRINGS "${NBL_BOOST_WAVE_DEP_FILE}" NBL_BOOST_LIBS)
	set(NBL_BOOST_LIBS ${NBL_BOOST_LIBS})
	list(POP_FRONT NBL_BOOST_LIBS)
	list(FILTER NBL_BOOST_LIBS EXCLUDE REGEX "#")
	list(FILTER NBL_BOOST_LIBS EXCLUDE REGEX "(unknown)")
	string(REPLACE "~" "/" NBL_BOOST_LIBS "${NBL_BOOST_LIBS}")

	# update boost .gitmodules configuration, discard all but modules reported by wave
	# NOTE: you commit this file to version control AND boost's .gitmodules *if got changed*,
	# use when updating boost to more recent version
	file(WRITE "${NBL_BOOST_WAVE_DEP_FILE}" "set(NBL_BOOST_LIBS ${NBL_BOOST_LIBS})")

	message(STATUS "Updating boost .gitmodules")
	foreach(SUBMODULE ${BOOST_SUBMODULES})
		# 1) fallback, ignore all
		NBL_BOOST_EXECUTE(git config --file .gitmodules submodule.${SUBMODULE}.update none)
	endforeach()

	foreach(NAME ${NBL_BOOST_LIBS})
		string(REPLACE "/" "_" SUBMODULE "${NAME}")
		message(STATUS "WAVE BOOST DEP SUBMODULE = ${SUBMODULE}")
		# 2) pick only submodules reported by wave
		NBL_BOOST_EXECUTE(git config --file .gitmodules submodule.${SUBMODULE}.update checkout)
	endforeach()
	# 3) and the top module itself
	NBL_BOOST_EXECUTE(git config --file .gitmodules submodule.wave.update checkout)
endif()

set(BOOST_PREPROCESSOR_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/superproject/libs/preprocessor/include" CACHE PATH "" FORCE)

get_filename_component(_BOOST_PREPROCESSOR_BR_BUNDLE_SEARCH_DIRECTORY_ "${BOOST_PREPROCESSOR_INCLUDE}" ABSOLUTE)
get_filename_component(_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/src" ABSOLUTE)
get_filename_component(_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)
set(BOOST_BUILTIN_RESOURCES_DIRECTORY_PATH "${_BOOST_PREPROCESSOR_BR_BUNDLE_SEARCH_DIRECTORY_}/boost" CACHE INTERNAL "" FORCE)

if(NBL_EMBED_BUILTIN_RESOURCES)
	include("${NBL_ROOT_PATH}/src/nbl/builtin/utils.cmake")
	
	file(GLOB_RECURSE BOOST_HEADERS_REC_REL RELATIVE "${BOOST_BUILTIN_RESOURCES_DIRECTORY_PATH}" "${BOOST_PREPROCESSOR_INCLUDE}/*")
	
	foreach(BOOST_HEADER_REL IN LISTS BOOST_HEADERS_REC_REL)
		LIST_BUILTIN_RESOURCE(BOOST_RESOURCES_TO_EMBED "${BOOST_HEADER_REL}")
	endforeach()

    ADD_CUSTOM_BUILTIN_RESOURCES(boostBuiltinResourceData BOOST_RESOURCES_TO_EMBED "${_BOOST_PREPROCESSOR_BR_BUNDLE_SEARCH_DIRECTORY_}" "boost" "boost::builtin" "${_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_HEADER_}" "${_BOOST_PREPROCESSOR_BR_OUTPUT_DIRECTORY_HEADER_}" "STATIC" "INTERNAL")
endif()

if(NOT EXISTS "${NBL_BOOST_WAVE_DEP_FILE}")
	message(FATAL_ERROR "Internal error, generate NBL_BOOST_WAVE_DEP_FILE by enabling NBL_BOOST_GENERATE_DEP_LIST!")
endif()

set(BOOST_STAGEDIR "${CMAKE_CURRENT_BINARY_DIR}/boost/superproject/stage")
include("${NBL_BOOST_WAVE_DEP_FILE}")

foreach(BOOST_LIB IN LISTS NBL_BOOST_LIBS)
	add_subdirectory(superproject/libs/${BOOST_LIB} EXCLUDE_FROM_ALL)
endforeach()

add_subdirectory(superproject/libs/wave EXCLUDE_FROM_ALL)

list(APPEND NBL_BOOST_TARGETS boost_wave) # wave
foreach(BOOST_LIB IN LISTS NBL_BOOST_LIBS)
	if(TARGET boost_${BOOST_LIB}) # wave's deps
		list(APPEND NBL_BOOST_TARGETS boost_${BOOST_LIB})
	endif()
endforeach()

set(NBL_BOOST_TARGETS 
	${NBL_BOOST_TARGETS}
PARENT_SCOPE)