include_guard(GLOBAL)

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
	foreach(_NBL_CONFIG_IMPL_ ${CMAKE_CONFIGURATION_TYPES})
		string(TOUPPER "${_NBL_CONFIG_IMPL_}" _NBL_CONFIG_U_IMPL_)
		list(APPEND _NBL_OPTIONS_IMPL_ MAP_${_NBL_CONFIG_U_IMPL_})
	endforeach()

	if(NOT _NBL_OPTIONS_IMPL_)
		message(FATAL_ERROR "Internal error, there are no configurations available! Please set \"CMAKE_CONFIGURATION_TYPES\"")
	endif()

	list(APPEND _NBL_OPTIONS_IMPL_ TARGET)
	cmake_parse_arguments(NBL "" "" "${_NBL_OPTIONS_IMPL_}" ${ARGN})
	
	# Profiles
	if(MSVC)
		include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/windows/msvc.cmake")
	elseif(ANDROID)
		include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/unix/android.cmake")
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/unix/gnu.cmake")
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/unix/clang.cmake")
	else()
		message(WARNING "UNTESTED COMPILER DETECTED, EXPECT WRONG OPTIMIZATION FLAGS! SUBMIT ISSUE ON GITHUB https://github.com/Devsh-Graphics-Programming/Nabla/issues")
	endif()

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
				set(NBL_${_NBL_OPTION_IMPL_}_ITEM 
					${NBL_${_NBL_OPTION_IMPL_}_ITEM}
				PARENT_SCOPE)
				
				# message("NBL_${_NBL_OPTION_IMPL_}[${_NBL_ARG_I_}]: ${NBL_${_NBL_OPTION_IMPL_}_ITEM}")
			endforeach()
			
			target_compile_options(${NBL_TARGET_ITEM} PUBLIC # the old behaviour was "PUBLIC" anyway, but we could also make it a param of the bundle call
				# global compile options
				${NBL_COMPILE_OPTIONS}
				
				# per configuration compile options with mapping
				$<$<CONFIG:${NBL_MAP_RELEASE_ITEM}>:${NBL_RELEASE_COMPILE_OPTIONS}>
				$<$<CONFIG:${NBL_MAP_DEBUG_ITEM}>:${NBL_DEBUG_COMPILE_OPTIONS}>
				$<$<CONFIG:${NBL_MAP_RELWITHDEBINFO_ITEM}>:${NBL_RELWITHDEBINFO_COMPILE_OPTIONS}>
			)
			
			math(EXPR _NBL_ARG_I_ "${_NBL_ARG_I_} + 1")
		endwhile()		
	else() # DIRECTORY mode
		list(REMOVE_ITEM _NBL_OPTIONS_IMPL_ TARGET)
		
		# global compile options
		add_compile_options(
			${NBL_COMPILE_OPTIONS}
		)
		
		foreach(_NBL_OPTION_IMPL_ ${_NBL_OPTIONS_IMPL_})
			string(REPLACE "NBL_MAP" "" NBL_MAP_CONFIGURATION_FROM "NBL_${_NBL_OPTION_IMPL_}")
			set(NBL_MAP_CONFIGURATION_TO ${NBL_${_NBL_OPTION_IMPL_}})
			set(NBL_TO_CONFIG_COMPILE_OPTIONS ${NBL_${NBL_MAP_CONFIGURATION_TO}_COMPILE_OPTIONS})
			
			# per configuration compile options with mapping
			add_compile_options(
				$<$<CONFIG:${NBL_MAP_CONFIGURATION_FROM}>:${NBL_TO_CONFIG_COMPILE_OPTIONS}>
			)
		endforeach()
	endif()
endfunction()