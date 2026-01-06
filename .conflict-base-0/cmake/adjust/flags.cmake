include_guard(GLOBAL)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

if(NOT DEFINED _NBL_JOBS_AMOUNT_)
	message(WARNING "\"${CMAKE_CURRENT_LIST_FILE}\" included without defined \"_NBL_JOBS_AMOUNT_\", setting it to \"1\"")
	set(_NBL_JOBS_AMOUNT_ 1)
endif()

define_property(TARGET PROPERTY NBL_CONFIGURATION_MAP
  BRIEF_DOCS "Stores configuration map for a target, it will evaluate to the configuration it's mapped to"
)

# https://github.com/Kitware/CMake/blob/05e77b8a27033e6fd086456bd6cef28338ff1474/Modules/Internal/CheckCompilerFlag.cmake#L26C7-L26C42
# must be cached because parse utility clears locals in the CheckCompilerFlag module
set(CHECK_COMPILER_FLAG_OUTPUT_VARIABLE NBL_COMPILER_FLAG_OUTPUT CACHE INTERNAL "")

# Usage: NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG <LANG;...> CONFIG <CONFIG;...> COMPILE_OPTIONS <OPTIONS;...> LINK_OPTIONS <OPTIONS;...>)
function(NBL_REQUEST_COMPILE_OPTION_SUPPORT)
	cmake_parse_arguments(IMPL "REQUIRED" "REQUEST_VAR" "LANG;CONFIG;COMPILE_OPTIONS;LINK_OPTIONS" ${ARGN})

	set(DEFAULT_COMPILERS c cxx)
	set(REQUEST_ALL_OPTIONS_PRESENT True)

	if(NOT IMPL_LANG)
        list(APPEND IMPL_LANG ${DEFAULT_COMPILERS})
    endif()

    foreach(COMPILER IN ITEMS ${IMPL_LANG})
        string(TOUPPER "${COMPILER}" COMPILER_UPPER)

		foreach(WHAT_OPTIONS IN ITEMS IMPL_COMPILE_OPTIONS IMPL_LINK_OPTIONS)
		    if(NOT ${WHAT_OPTIONS})
				continue()
			endif()

			set(IMPL_OPTIONS ${${WHAT_OPTIONS}})
			string(REPLACE IMPL_ "" WHAT_OPTIONS "${WHAT_OPTIONS}")

			foreach(COMPILE_OPTION ${IMPL_OPTIONS})
				if(IMPL_CONFIG)
					foreach(CONFIG ${IMPL_CONFIG})
						# TODO: validate (${CONFIG} \in ${CMAKE_CONFIGURATION_TYPES})
						string(TOUPPER "${CONFIG}" CONFIG_UPPER)
						set(NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_${WHAT_OPTIONS} "${NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_${WHAT_OPTIONS}};${COMPILE_OPTION}")
					endforeach()
				else()
					set(NBL_${COMPILER_UPPER}_${WHAT_OPTIONS} "${NBL_${COMPILER_UPPER}_${WHAT_OPTIONS}};${COMPILE_OPTION}")
				endif()
			endforeach()

			if(IMPL_CONFIG)
				foreach(CONFIG ${IMPL_CONFIG})
					string(TOUPPER "${CONFIG}" CONFIG_UPPER)
					set(NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_${WHAT_OPTIONS} ${NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_${WHAT_OPTIONS}} PARENT_SCOPE)
				endforeach()
			else()
				set(NBL_${COMPILER_UPPER}_${WHAT_OPTIONS} ${NBL_${COMPILER_UPPER}_${WHAT_OPTIONS}} PARENT_SCOPE)
			endif()
		endforeach()
    endforeach()
endfunction()

option(NBL_REQUEST_SSE_4_2 "Request compilation with SSE 4.2 instruction set enabled for Nabla projects" ON)
option(NBL_REQUEST_SSE_AVX2 "Request compilation with SSE Intel Advanced Vector Extensions 2 for Nabla projects" ON)

# profiles
foreach(NBL_COMPILER_LANGUAGE IN ITEMS C CXX)
    # all list of all known by CMake vendors:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_ID.html
    set(NBL_COMPILER_VENDOR "${CMAKE_${NBL_COMPILER_LANGUAGE}_COMPILER_ID}")
    set(NBL_PROFILE_NAME "${NBL_COMPILER_LANGUAGE}_${NBL_COMPILER_VENDOR}") # eg. "cxx_MSVC.cmake"
    set(NBL_PROFILE_PATH "${CMAKE_CURRENT_LIST_DIR}/template/vendor/${NBL_PROFILE_NAME}.cmake")

    include("${NBL_PROFILE_PATH}" RESULT_VARIABLE _NBL_FOUND_)

    if(NOT _NBL_FOUND_)
        message(WARNING "UNSUPPORTED \"${NBL_COMPILER_LANGUAGE}\" COMPILER LANGUAGE FOR \"${NBL_COMPILER_VENDOR}\" DETECTED, CMAKE CONFIGURATION OR BUILD MAY FAIL AND COMPILE OPTIONS FLAGS WILL NOT BE SET! SUBMIT ISSUE ON GITHUB https://github.com/Devsh-Graphics-Programming/Nabla/issues")
        continue()
    endif()

	# a profile MUST define 

    # - "NBL_${NBL_COMPILER_LANGUAGE}_${CONFIGURATION}_${WHAT}_OPTIONS" (configuration dependent)
    # - "NBL_${NBL_COMPILER_LANGUAGE}_${WHAT}_OPTIONS" (global)

	# a profile MUST NOT define
		# - NBL_${WHAT}_OPTIONS
		
	# note: 
	# - use NBL_REQUEST_COMPILE_OPTION_SUPPORT in profile to creates those vars
	# - include reset utility in profiles to init vars with empty lists

	# TODO: DEFINITIONS for WHAT to unify the API

	foreach(WHAT COMPILE LINK)
		set(NBL_OPTIONS_VAR_NAME NBL_${NBL_COMPILER_LANGUAGE}_${WHAT}_OPTIONS)
		set(NBL_OPTIONS_VAR_VALUE ${${NBL_OPTIONS_VAR_NAME}})

		if(NOT DEFINED ${NBL_OPTIONS_VAR_NAME})
			message(FATAL_ERROR "\"${NBL_PROFILE_PATH}\" did not define \"${NBL_OPTIONS_VAR_NAME}\"!")
		endif()

		# update map with configuration dependent compile options
		foreach(CONFIGURATION IN ITEMS RELEASE RELWITHDEBINFO DEBUG)
			set(NBL_CONFIGURATION_OPTIONS_VAR_NAME NBL_${NBL_COMPILER_LANGUAGE}_${CONFIGURATION}_${WHAT}_OPTIONS)
			set(NBL_CONFIGURATION_OPTIONS_VAR_VALUE ${${NBL_CONFIGURATION_OPTIONS_VAR_NAME}})

			if(NOT DEFINED ${NBL_CONFIGURATION_OPTIONS_VAR_NAME})
				message(FATAL_ERROR "\"${NBL_PROFILE_PATH}\" did not define \"${NBL_CONFIGURATION_OPTIONS_VAR_NAME}\"!")
			endif()

			set(NBL_${CONFIGURATION}_${WHAT}_OPTIONS ${NBL_${CONFIGURATION}_${WHAT}_OPTIONS}
				# note that "${NBL_CONFIGURATION_OPTIONS_VAR_VALUE}" MUST NOT contain ANY 
				# $<$<CONFIG:<>> generator expression in order to support our configuration mapping features
				$<$<${WHAT}_LANGUAGE:${NBL_COMPILER_LANGUAGE}>:${NBL_CONFIGURATION_OPTIONS_VAR_VALUE}>
			)
		endforeach()

		# update map with global compile options
		set(NBL_${WHAT}_OPTIONS ${NBL_${WHAT}_OPTIONS}
			$<$<${WHAT}_LANGUAGE:${NBL_COMPILER_LANGUAGE}>:${NBL_${NBL_COMPILER_LANGUAGE}_${WHAT}_OPTIONS}>
		)
	endforeach()

	block()
		# validate build with a vendor profile, any warning diagnostic = error
		# if you hit error it means the profile generates diagnostics due to:
		# - an option (compile or link) which doesn't exist (typo? check vendor docs)
		# - a set of options which invalidates an option (eg. MSVC's /INCREMENTAL with /LTCG:incremental is invalid, however linker will emit a warning by default + do a fall-back)
		# https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_FLAGS.html#variable:CMAKE_%3CLANG%3E_FLAGS
		# https://cmake.org/cmake/help/latest/module/CheckCompilerFlag.html#command:check_compiler_flag

		set(CMAKE_${NBL_COMPILER_LANGUAGE}_FLAGS)

		foreach(CONFIGURATION IN ITEMS Release RelWithDebInfo Debug)
			set(CMAKE_TRY_COMPILE_CONFIGURATION ${CONFIGURATION})
			string(TOUPPER "${CONFIGURATION}" CONFIGURATION)

			set(TEST_NAME "NBL_${NBL_COMPILER_LANGUAGE}_LANG_${CONFIGURATION}_BUILD_OPTIONS_SUPPORT")
			set(CMAKE_${NBL_COMPILER_LANGUAGE}_FLAGS_${CONFIGURATION})

			set(COMPILE_OPTIONS ${NBL_${NBL_COMPILER_LANGUAGE}_COMPILE_OPTIONS} ${NBL_${NBL_COMPILER_LANGUAGE}_${CONFIGURATION}_COMPILE_OPTIONS})
			set(LINK_OPTIONS ${NBL_${NBL_COMPILER_LANGUAGE}_${CONFIGURATION}_LINK_OPTIONS})
			set(COMBINED ${COMPILE_OPTIONS} ${LINK_OPTIONS})

			set(NBL_OUTPUT_FILE "${CMAKE_BINARY_DIR}/.nbl/try-compile/${TEST_NAME}.output") # no hash in output diagnostic file, desired
			
			string(SHA1 OPTIONS_HASH "${COMBINED}")
			string(APPEND TEST_NAME "_HASH_${OPTIONS_HASH}")

			set(FLAG_VAR ${TEST_NAME})
			set(CMAKE_REQUIRED_LINK_OPTIONS ${LINK_OPTIONS})
			string(REPLACE ";" " " CLI_COMPILE_OPTIONS "${COMPILE_OPTIONS}")

			if(NBL_COMPILER_LANGUAGE STREQUAL C)
				check_c_compiler_flag("${CLI_COMPILE_OPTIONS}" "${FLAG_VAR}")
			elseif(NBL_COMPILER_LANGUAGE STREQUAL CXX)
				check_cxx_compiler_flag("${CLI_COMPILE_OPTIONS}" "${FLAG_VAR}")
			endif()

			if(NOT ${FLAG_VAR})
				if(NOT "${NBL_COMPILER_FLAG_OUTPUT}" STREQUAL "")
					file(WRITE "${NBL_OUTPUT_FILE}" "${NBL_COMPILER_FLAG_OUTPUT}") # lock into file, do not cache, must read from the file because of NBL_COMPILER_FLAG_OUTPUT availability (CMake module writes an output only once before a signature flag status is created)
				endif()

				if(EXISTS "${NBL_OUTPUT_FILE}")
					file(READ "${NBL_OUTPUT_FILE}" NBL_DIAGNOSTICS)
					set(NBL_DIAGNOSTICS "Diagnostics:\n${NBL_DIAGNOSTICS}")
				else()
					set(NBL_DIAGNOSTICS)
				endif()

				if(NOT DEFINED NBL_SKIP_BUILD_OPTIONS_VALIDATION)
					message(FATAL_ERROR "${TEST_NAME} failed! To skip the validation define \"NBL_SKIP_BUILD_OPTIONS_VALIDATION\". ${NBL_DIAGNOSTICS}")
				endif()
			endif()
		endforeach()
	endblock()
endforeach()

function(NBL_EXT_P_APPEND_COMPILE_OPTIONS NBL_LIST_NAME MAP_RELEASE MAP_RELWITHDEBINFO MAP_DEBUG)		
	macro(NBL_MAP_CONFIGURATION NBL_CONFIG_FROM NBL_CONFIG_TO)
		string(TOUPPER "${NBL_CONFIG_FROM}" NBL_CONFIG_FROM_U)
		string(TOUPPER "${NBL_CONFIG_TO}" NBL_CONFIG_TO_U)
		
		string(REPLACE ";" " " _NBL_CXX_CO_ "${NBL_CXX_${NBL_CONFIG_TO_U}_COMPILE_OPTIONS}")
		string(REPLACE ";" " " _NBL_C_CO_ "${NBL_C_${NBL_CONFIG_TO_U}_COMPILE_OPTIONS}")
		
		list(APPEND ${NBL_LIST_NAME} "-DCMAKE_CXX_FLAGS_${NBL_CONFIG_FROM_U}:STRING=${_NBL_CXX_CO_}")
		list(APPEND ${NBL_LIST_NAME} "-DCMAKE_C_FLAGS_${NBL_CONFIG_FROM_U}:STRING=${_NBL_C_CO_}")
	endmacro()
	
	NBL_MAP_CONFIGURATION(RELEASE ${MAP_RELEASE})
	NBL_MAP_CONFIGURATION(RELWITHDEBINFO ${MAP_RELWITHDEBINFO})
	NBL_MAP_CONFIGURATION(DEBUG ${MAP_DEBUG})
	
	set(${NBL_LIST_NAME} 
		${${NBL_LIST_NAME}}
	PARENT_SCOPE)
endfunction()

# Adjust compile flags for the build system, supports calling per target or directory and map a configuration into another one.
#
# -- TARGET mode --
#
# nbl_adjust_flags(
#	TARGET <NAME_OF_TARGET> MAP_RELEASE <CONFIGURATION> MAP_RELWITHDEBINFO <CONFIGURATION> MAP_DEBUG <CONFIGURATION>
#	...
#	TARGET <NAME_OF_TARGET> MAP_RELEASE <CONFIGURATION> MAP_RELWITHDEBINFO <CONFIGURATION> MAP_DEBUG <CONFIGURATION>
# )
#
# -- DIRECTORY mode --
#
# nbl_adjust_flags(
#	MAP_RELEASE <CONFIGURATION> MAP_RELWITHDEBINFO <CONFIGURATION> MAP_DEBUG <CONFIGURATION>
# )

function(nbl_adjust_flags)
	# only configuration dependent, global CMAKE_<LANG>_FLAGS flags are fine
	macro(UNSET_GLOBAL_CONFIGURATION_FLAGS NBL_CONFIGURATION)
		if(DEFINED CMAKE_CXX_FLAGS_${NBL_CONFIGURATION})
			unset(CMAKE_CXX_FLAGS_${NBL_CONFIGURATION} CACHE)
		endif()
		
		if(DEFINED CMAKE_C_FLAGS_${NBL_CONFIGURATION})
			unset(CMAKE_C_FLAGS_${NBL_CONFIGURATION} CACHE)
		endif()
	endmacro()

	foreach(_NBL_CONFIG_IMPL_ ${CMAKE_CONFIGURATION_TYPES})
		string(TOUPPER "${_NBL_CONFIG_IMPL_}" _NBL_CONFIG_U_IMPL_)
		UNSET_GLOBAL_CONFIGURATION_FLAGS(${_NBL_CONFIG_U_IMPL_})
		
		list(APPEND _NBL_OPTIONS_IMPL_ MAP_${_NBL_CONFIG_U_IMPL_})
	endforeach()

	if(NOT _NBL_OPTIONS_IMPL_)
		message(FATAL_ERROR "Internal error, there are no configurations available! Please set \"CMAKE_CONFIGURATION_TYPES\"")
	endif()

	list(APPEND _NBL_OPTIONS_IMPL_ TARGET)
	cmake_parse_arguments(NBL "" "" "${_NBL_OPTIONS_IMPL_}" ${ARGN})

	# TARGET mode
	if(NBL_TARGET)
		# validate	
		list(LENGTH NBL_TARGET _NBL_V_OPTION_LEN_)
		list(REMOVE_ITEM _NBL_OPTIONS_IMPL_ TARGET)
		foreach(_NBL_OPTION_IMPL_ ${_NBL_OPTIONS_IMPL_})
			if(NOT NBL_${_NBL_OPTION_IMPL_})
				message(FATAL_ERROR "Internal error, nbl_adjust_flags called with TARGET mode missing \"${_NBL_OPTION_IMPL_}\" argument!")
			endif()
			
			list(LENGTH NBL_${_NBL_OPTION_IMPL_} _NBL_C_V_OPTION_LEN_)
			if("${_NBL_C_V_OPTION_LEN_}" STREQUAL "${_NBL_V_OPTION_LEN_}")
				set(_NBL_V_OPTION_LEN_ 
					"${_NBL_C_V_OPTION_LEN_}"
				PARENT_SCOPE)
			else()
				message(FATAL_ERROR "Internal error, nbl_adjust_flags called with TARGET mode has inequal arguments!")
			endif()
		endforeach()
		list(APPEND _NBL_OPTIONS_IMPL_ TARGET)
		
		set(_NBL_ARG_I_ 0)
		while(_NBL_ARG_I_ LESS ${_NBL_V_OPTION_LEN_})		
			foreach(_NBL_OPTION_IMPL_ ${_NBL_OPTIONS_IMPL_})
				list(GET NBL_${_NBL_OPTION_IMPL_} ${_NBL_ARG_I_} NBL_${_NBL_OPTION_IMPL_}_ITEM)
				string(TOUPPER "${NBL_${_NBL_OPTION_IMPL_}_ITEM}" NBL_${_NBL_OPTION_IMPL_}_ITEM_U)
				
				set(NBL_${_NBL_OPTION_IMPL_}_ITEM 
					${NBL_${_NBL_OPTION_IMPL_}_ITEM}
				PARENT_SCOPE)
				
				set(NBL_${_NBL_OPTION_IMPL_}_ITEM_U 
					${NBL_${_NBL_OPTION_IMPL_}_ITEM_U}
				PARENT_SCOPE)
			endforeach()

			# global compile options
			list(APPEND _D_NBL_COMPILE_OPTIONS_ ${NBL_COMPILE_OPTIONS})

			foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
				string(TOUPPER "${CONFIG}" CONFIG_U)

				# per configuration options with mapping
				foreach(WHAT COMPILE LINK)
					list(APPEND _D_NBL_${WHAT}_OPTIONS_ $<$<CONFIG:${CONFIG}>:${NBL_${NBL_MAP_${CONFIG_U}_ITEM_U}_${WHAT}_OPTIONS}>)
				endforeach()

				# configuration mapping properties
				string(APPEND _D_NBL_CONFIGURATION_MAP_ $<$<CONFIG:${CONFIG}>:${NBL_MAP_${CONFIG_U}_ITEM_U}>)
			endforeach()
			
			set_target_properties(${NBL_TARGET_ITEM} PROPERTIES
				NBL_CONFIGURATION_MAP "${_D_NBL_CONFIGURATION_MAP_}"
				COMPILE_OPTIONS "${_D_NBL_COMPILE_OPTIONS_}"
				LINK_OPTIONS "${_D_NBL_LINK_OPTIONS_}"
			)
			unset(_D_NBL_CONFIGURATION_MAP_)
			unset(_D_NBL_COMPILE_OPTIONS_)
			unset(_D_NBL_LINK_OPTIONS_)
			
			set(MAPPED_CONFIG $<TARGET_GENEX_EVAL:${NBL_TARGET_ITEM},$<TARGET_PROPERTY:${NBL_TARGET_ITEM},NBL_CONFIGURATION_MAP>>)
			
			set_target_properties(${NBL_TARGET_ITEM} PROPERTIES
				MSVC_DEBUG_INFORMATION_FORMAT $<$<OR:$<STREQUAL:${MAPPED_CONFIG},DEBUG>,$<STREQUAL:${MAPPED_CONFIG},RELWITHDEBINFO>>:ProgramDatabase> # ignored on non xMSVC-ABI targets
			)

			math(EXPR _NBL_ARG_I_ "${_NBL_ARG_I_} + 1")
		endwhile()		
	else() # DIRECTORY mode
		list(REMOVE_ITEM _NBL_OPTIONS_IMPL_ TARGET)
		
		# global compile options
		list(APPEND _D_NBL_COMPILE_OPTIONS_ ${NBL_COMPILE_OPTIONS})
		foreach(_NBL_OPTION_IMPL_ ${_NBL_OPTIONS_IMPL_})
			string(REPLACE "NBL_MAP_" "" NBL_MAP_CONFIGURATION_FROM "NBL_${_NBL_OPTION_IMPL_}")
			string(TOUPPER "${NBL_${_NBL_OPTION_IMPL_}}" NBL_MAP_CONFIGURATION_TO)
			set(NBL_TO_CONFIG_COMPILE_OPTIONS ${NBL_${NBL_MAP_CONFIGURATION_TO}_COMPILE_OPTIONS})
			
			# per configuration compile options with mapping
			list(APPEND _D_NBL_COMPILE_OPTIONS_ $<$<CONFIG:${NBL_MAP_CONFIGURATION_FROM}>:${NBL_TO_CONFIG_COMPILE_OPTIONS}>)
		endforeach()
		
		set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_D_NBL_COMPILE_OPTIONS_}")
	endif()
endfunction()