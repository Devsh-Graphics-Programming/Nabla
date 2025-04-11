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

# Usage: NBL_REQUEST_COMPILE_OPTION_SUPPORT(LANG <LANG;...> CONFIG <CONFIG;...> OPTIONS <OPTIONS;...> )
# LANG, CONFIG - optional, OPTIONS - required
function(NBL_REQUEST_COMPILE_OPTION_SUPPORT)
	cmake_parse_arguments(IMPL "" "REQUEST_VAR;REQUIRED" "LANG;CONFIG;OPTIONS" ${ARGN})

	set(DEFAULT_COMPILERS c cxx)
	set(REQUEST_ALL_OPTIONS_PRESENT True)

	if(NOT IMPL_LANG)
        list(APPEND IMPL_LANG ${DEFAULT_COMPILERS})
    endif()

    if(NOT IMPL_OPTIONS)
        message(FATAL_ERROR "NBL_REQUEST_COMPILE_OPTION_SUPPORT's OPTIONS empty!")
    endif()

    foreach(COMPILER IN ITEMS ${IMPL_LANG})
        string(TOUPPER "${COMPILER}" COMPILER_UPPER)

		if(COMPILER_UPPER STREQUAL C)
			macro(VALIDATE_FLAG) 
				check_c_compiler_flag(${ARGV})
			endmacro()
		elseif(COMPILER_UPPER STREQUAL CXX)
			macro(VALIDATE_FLAG) 
				check_cxx_compiler_flag(${ARGV})
			endmacro()
		endif()

		foreach(COMPILE_OPTION ${IMPL_OPTIONS})
			string(REGEX REPLACE "[-=:;/.]" "_" FLAG_SIGNATURE "${COMPILE_OPTION}")
			set(FLAG_VAR "NBL_${COMPILER_UPPER}_COMPILER_HAS_${FLAG_SIGNATURE}_FLAG")

			VALIDATE_FLAG("${COMPILE_OPTION}" "${FLAG_VAR}")

			if(${FLAG_VAR})
				if(IMPL_CONFIG)
					foreach(CONFIG ${IMPL_CONFIG})
						# TODO: validate (${CONFIG} \in ${CMAKE_CONFIGURATION_TYPES})
						string(TOUPPER "${CONFIG}" CONFIG_UPPER)
						set(NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_COMPILE_OPTIONS "${NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_COMPILE_OPTIONS};${COMPILE_OPTION}")
					endforeach()
				else()
					set(NBL_${COMPILER_UPPER}_COMPILE_OPTIONS "${NBL_${COMPILER_UPPER}_COMPILE_OPTIONS};${COMPILE_OPTION}")
				endif()
			else()
				if(IMPL_REQUIRED)
					message(FATAL_ERROR "Terminating, NBL_REQUEST_COMPILE_OPTION_SUPPORT was invoked with REQUIRED qualifier!")
				endif()

				set(REQUEST_ALL_OPTIONS_PRESENT False)
			endif()
		endforeach()

		if(IMPL_CONFIG)
			foreach(CONFIG ${IMPL_CONFIG})
				string(TOUPPER "${CONFIG}" CONFIG_UPPER)
				set(NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_COMPILE_OPTIONS ${NBL_${COMPILER_UPPER}_${CONFIG_UPPER}_COMPILE_OPTIONS} PARENT_SCOPE)
			endforeach()
		else()
			set(NBL_${COMPILER_UPPER}_COMPILE_OPTIONS ${NBL_${COMPILER_UPPER}_COMPILE_OPTIONS} PARENT_SCOPE)
		endif()
    endforeach()

	if(IMPL_REQUEST_VAR)
		set(${IMPL_REQUEST_VAR} ${REQUEST_ALL_OPTIONS_PRESENT} PARENT_SCOPE)
	endif()
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
        # - "NBL_${NBL_COMPILER_LANGUAGE}_${CONFIGURATION}_COMPILE_OPTIONS" (configuration dependent)
        # - "NBL_${NBL_COMPILER_LANGUAGE}_COMPILE_OPTIONS" (global)

    # a profile MUST NOT define
        # - NBL_COMPILE_OPTIONS

    set(NBL_COMPILE_OPTIONS_VAR_NAME NBL_${NBL_COMPILER_LANGUAGE}_COMPILE_OPTIONS)
    set(NBL_COMPILE_OPTIONS_VAR_VALUE ${${NBL_COMPILE_OPTIONS_VAR_NAME}})

    if(NOT DEFINED ${NBL_COMPILE_OPTIONS_VAR_NAME})
        message(FATAL_ERROR "\"${NBL_PROFILE_PATH}\" did not define \"${NBL_COMPILE_OPTIONS_VAR_NAME}\"!")
    endif()

    # update map with configuration dependent compile options
    foreach(CONFIGURATION IN ITEMS RELEASE RELWITHDEBINFO DEBUG)
        set(NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_NAME NBL_${NBL_COMPILER_LANGUAGE}_${CONFIGURATION}_COMPILE_OPTIONS)
        set(NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_VALUE ${${NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_NAME}})

        if(NOT DEFINED ${NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_NAME})
            message(FATAL_ERROR "\"${NBL_PROFILE_PATH}\" did not define \"${NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_NAME}\"!")
        endif()

        list(APPEND NBL_${CONFIGURATION}_COMPILE_OPTIONS
            # note that "${NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_VALUE}" MUST NOT contain ANY 
            # $<$<CONFIG:<>> generator expression in order to support our configuration mapping features
            $<$<COMPILE_LANGUAGE:${NBL_COMPILER_LANGUAGE}>:${NBL_CONFIGURATION_COMPILE_OPTIONS_VAR_VALUE}>
        )

        set(NBL_${CONFIGURATION}_COMPILE_OPTIONS  ${NBL_${CONFIGURATION}_COMPILE_OPTIONS})
    endforeach()

    # update map with global compile options
    list(APPEND NBL_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:${NBL_COMPILER_LANGUAGE}>:${NBL_${NBL_COMPILER_LANGUAGE}_COMPILE_OPTIONS}>)

    set(NBL_COMPILE_OPTIONS ${NBL_COMPILE_OPTIONS})
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
			
			# per configuration compile options with mapping
			list(APPEND _D_NBL_COMPILE_OPTIONS_ $<$<CONFIG:Debug>:${NBL_${NBL_MAP_DEBUG_ITEM_U}_COMPILE_OPTIONS}>)
			list(APPEND _D_NBL_COMPILE_OPTIONS_ $<$<CONFIG:Release>:${NBL_${NBL_MAP_RELEASE_ITEM_U}_COMPILE_OPTIONS}>)
			list(APPEND _D_NBL_COMPILE_OPTIONS_ $<$<CONFIG:RelWithDebInfo>:${NBL_${NBL_MAP_RELWITHDEBINFO_ITEM_U}_COMPILE_OPTIONS}>)
			
			# configuration mapping properties
			string(APPEND _D_NBL_CONFIGURATION_MAP_ $<$<CONFIG:Debug>:${NBL_MAP_DEBUG_ITEM_U}>)
			string(APPEND _D_NBL_CONFIGURATION_MAP_ $<$<CONFIG:Release>:${NBL_MAP_RELEASE_ITEM_U}>)
			string(APPEND _D_NBL_CONFIGURATION_MAP_ $<$<CONFIG:RelWithDebInfo>:${NBL_MAP_RELWITHDEBINFO_ITEM_U}>)
			
			set_target_properties(${NBL_TARGET_ITEM} PROPERTIES
				NBL_CONFIGURATION_MAP "${_D_NBL_CONFIGURATION_MAP_}"
				COMPILE_OPTIONS "${_D_NBL_COMPILE_OPTIONS_}"
			)
			unset(_D_NBL_CONFIGURATION_MAP_)
			unset(_D_NBL_COMPILE_OPTIONS_)
			
			set(MAPPED_CONFIG $<TARGET_GENEX_EVAL:${NBL_TARGET_ITEM},$<TARGET_PROPERTY:${NBL_TARGET_ITEM},NBL_CONFIGURATION_MAP>>)
			
			if(CMAKE_CXX_COMPILER_ID STREQUAL MSVC)
				if(NBL_SANITIZE_ADDRESS)
					set(NBL_TARGET_MSVC_DEBUG_INFORMATION_FORMAT "$<$<OR:$<STREQUAL:${MAPPED_CONFIG},DEBUG>,$<STREQUAL:${MAPPED_CONFIG},RELWITHDEBINFO>>:ProgramDatabase>")
				else()
					set(NBL_TARGET_MSVC_DEBUG_INFORMATION_FORMAT "$<$<STREQUAL:${MAPPED_CONFIG},DEBUG>:EditAndContinue>$<$<STREQUAL:${MAPPED_CONFIG},RELWITHDEBINFO>:ProgramDatabase>")
				endif()	
			endif()
			
			set_target_properties(${NBL_TARGET_ITEM} PROPERTIES
				MSVC_DEBUG_INFORMATION_FORMAT "${NBL_TARGET_MSVC_DEBUG_INFORMATION_FORMAT}"
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