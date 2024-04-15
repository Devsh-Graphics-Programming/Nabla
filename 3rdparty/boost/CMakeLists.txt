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

get_filename_component(NBL_BOOST_WAVE_DEP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/dep/wave.cmake" ABSOLUTE)

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

get_target_property(BOOST_WAVE_BINARY_DIR boost_wave BINARY_DIR)
get_filename_component(BOOST_WAVE_BINARY_DIR "${BOOST_WAVE_BINARY_DIR}" ABSOLUTE)

list(APPEND BOOST_WAVE_LINK_LIBRARIES # hardcode it, avoid recursive traversal through targets
	boost_wave
	boost_filesystem
	boost_serialization
	boost_thread
	boost_atomic
	boost_chrono
	boost_date_time
	boost_container
	boost_exception
)

string(APPEND _BLL_GE_CONTENT_ "set(NBL_BOOST_LIBRARIES\n")
foreach(_BLL_ IN LISTS BOOST_WAVE_LINK_LIBRARIES)
	get_target_property(ITYPE ${_BLL_} TYPE)
	
	if ("${_FOUND_}" STREQUAL "-1")
		message(FATAL_ERROR "Internal error, _BLL_ = ${_BLL_} is ${ITYPE} type, must be STATIC_LIBRARY")
	endif()
	
	string(APPEND _BLL_GE_CONTENT_ "$<TARGET_FILE:${_BLL_}>\n")
endforeach()
string(APPEND _BLL_GE_CONTENT_ ")")

set(OF_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/ext/release/libraries.cmake")
file(GENERATE OUTPUT "${OF_OUTPUT_PATH}" CONTENT "${_BLL_GE_CONTENT_}" CONDITION $<CONFIG:Release>)
include("${OF_OUTPUT_PATH}")

ExternalProject_Add(boost
	DOWNLOAD_COMMAND  ""
	SOURCE_DIR        "${CMAKE_CURRENT_SOURCE_DIR}/superproject/libs/wave" # doesnt matter
	BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/ext"
	CONFIGURE_COMMAND "" # proxy
	BUILD_COMMAND 	  "${CMAKE_COMMAND}" --build "${BOOST_WAVE_BINARY_DIR}" --config Release
	BUILD_ALWAYS 	  TRUE
	CONFIGURE_HANDLED_BY_BUILD FALSE
	INSTALL_COMMAND	  ""
	TEST_COMMAND      ""
)

set(NBL_BOOST_TARGETS 
	${NBL_BOOST_TARGETS}
PARENT_SCOPE)

set(NBL_BOOST_LIBRARIES
	${NBL_BOOST_LIBRARIES}
PARENT_SCOPE)

# Boost uses it's own tool for generating dependency list for targets, therefore we 
# can make sure manually added dependnecy subdirectories for a library are valid
# https://www.boost.org/doc/libs/1_83_0/tools/boostdep/doc/html/index.html#boostdep.introduction.building_boostdep

if(NBL_BOOST_GENERATE_DEP_LIST) # internal, for Nabla devs
	if(WIN32)
		set(NBL_BOOSTDEP_EXE "boostdep.exe")
	else()
		set(NBL_BOOSTDEP_EXE "boostdep")
	endif()
	
	set(NBL_BOOSTDEP_EXE_FILEPATH "${CMAKE_CURRENT_BINARY_DIR}/superproject/tools/boostdep/bin/${NBL_BOOSTDEP_EXE}")
	
	if(NOT EXISTS "${NBL_BOOSTDEP_EXE_FILEPATH}")
		macro(NBL_BOOST_EXECUTE)
			execute_process(COMMAND ${ARGV}
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/superproject"
			)
		endmacro()

		NBL_BOOST_EXECUTE(cmd /C bootstrap.bat)
		NBL_BOOST_EXECUTE(cmd /C b2.exe tools/boostdep/build)
		NBL_BOOST_EXECUTE("${CMAKE_COMMAND}" -E copy "./dist/bin/${NBL_BOOSTDEP_EXE}" "${NBL_BOOSTDEP_EXE_FILEPATH}")
		NBL_BOOST_EXECUTE(git clean -fdx)
		NBL_BOOST_EXECUTE(git reset --hard)
	endif()

	execute_process(COMMAND "${NBL_BOOSTDEP_EXE_FILEPATH}" --boost-root "${CMAKE_CURRENT_SOURCE_DIR}/superproject" --brief wave
		OUTPUT_VARIABLE NBL_OUTPUT_VAR
	)

	file(WRITE "${NBL_BOOST_WAVE_DEP_FILE}" "${NBL_OUTPUT_VAR}")

	file(STRINGS "${NBL_BOOST_WAVE_DEP_FILE}" NBL_BOOST_LIBS)
	set(NBL_BOOST_LIBS ${NBL_BOOST_LIBS})
	list(POP_FRONT NBL_BOOST_LIBS)
	list(FILTER NBL_BOOST_LIBS EXCLUDE REGEX "#")
	list(FILTER NBL_BOOST_LIBS EXCLUDE REGEX "(unknown)")
	string(REPLACE "~" "/" NBL_BOOST_LIBS "${NBL_BOOST_LIBS}")

	file(WRITE "${NBL_BOOST_WAVE_DEP_FILE}" "set(NBL_BOOST_LIBS ${NBL_BOOST_LIBS})")
endif()
