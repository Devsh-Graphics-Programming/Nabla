# Assigns builtin resources to a bundle a target library will be created with
# _BUNDLE_NAME_ is a bundle name, must be a valid CMake variable
# _LBR_PATH_ is a path to builtin resource

macro(LIST_BUILTIN_RESOURCE _BUNDLE_NAME_ _LBR_PATH_)
	math(EXPR _ALIAS_C_ "${ARGC} - 2" OUTPUT_FORMAT DECIMAL)
	set(_ALIAS_ARGS_ ${ARGV})
	
	if("${_ALIAS_C_}" GREATER "0")
		list(SUBLIST _ALIAS_ARGS_ "2" "${_ALIAS_C_}" _ALIAS_ARGS_)
		
		foreach(_ALIAS_ IN LISTS _ALIAS_ARGS_)
			string(APPEND _OPTIONAL_ALIASES_ ",${_ALIAS_}")
		endforeach()
	endif()
	
	list(APPEND ${_BUNDLE_NAME_} "${_LBR_PATH_}")
	set(${_BUNDLE_NAME_} ${${_BUNDLE_NAME_}} PARENT_SCOPE) # override
	
	list(APPEND _LBR_${_BUNDLE_NAME_}_ "${_LBR_PATH_}${_OPTIONAL_ALIASES_}")
	set(_LBR_${_BUNDLE_NAME_}_ ${_LBR_${_BUNDLE_NAME_}_} PARENT_SCOPE) # override
	
	unset(_OPTIONAL_ALIASES_)
	unset(_ALIAS_ARGS_)
endmacro()

# Creates a library with builtin resources for given bundle
# _TARGET_NAME_ is name of a target library that will be created
# _BUNDLE_NAME_ is a bundle name builtin resources are associated with
# _BUNDLE_SEARCH_DIRECTORY_ is an absolute search directory path for builtin resorces for given bundle
# _NAMESPACE_ is a C++ namespace builtin resources will be wrapped into
# _OUTPUT_INCLUDE_SEARCH_DIRECTORY_ is an absolute path to output directory for builtin resources header files which will be a search directory for generated headers outputed to ${_OUTPUT_HEADER_DIRECTORY_}/${_PATH_PREFIX_} where path prefix is the namespace turned into a path
# _OUTPUT_SOURCE_DIRECTORY_ is an absolute path to output directory for builtin resources source files

function(ADD_CUSTOM_BUILTIN_RESOURCES _TARGET_NAME_ _BUNDLE_NAME_ _BUNDLE_SEARCH_DIRECTORY_ _NAMESPACE_ _OUTPUT_INCLUDE_SEARCH_DIRECTORY_ _OUTPUT_SOURCE_DIRECTORY_)
	if(NOT DEFINED _Python3_EXECUTABLE)
		message(FATAL_ERROR "_Python3_EXECUTABLE must be defined - call find_package(Python3 COMPONENTS Interpreter REQUIRED)")
	endif()

	set(NBL_TEMPLATE_RESOURCES_ARCHIVE_HEADER "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/CArchive.h.in")
	set(NBL_BUILTIN_HEADER_GEN_PY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/builtinHeaderGen.py")
	set(NBL_BUILTIN_DATA_GEN_PY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/builtinDataGen.py")
	set(NBL_BS_HEADER_FILENAME "builtinResources.h")
	set(NBL_BS_DATA_SOURCE_FILENAME "builtinResourceData.cpp")
	
	string(REPLACE "::" "/" _PATH_PREFIX_ "${_NAMESPACE_}")
	string(REPLACE "::" "_" _GUARD_SUFFIX_ "${_NAMESPACE_}")
	string(REGEX REPLACE "^[0-9]+\." "" _GUARD_SUFFIX_ ${_GUARD_SUFFIX_})
	string(TOUPPER ${_GUARD_SUFFIX_} _GUARD_SUFFIX_)
	string(MAKE_C_IDENTIFIER ${_GUARD_SUFFIX_} _GUARD_SUFFIX_)
	
	set(_OUTPUT_INCLUDE_SEARCH_DIRECTORY_ "${_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}")
	set(_OUTPUT_HEADER_DIRECTORY_ "${_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}/${_PATH_PREFIX_}")
	
	file(MAKE_DIRECTORY "${_OUTPUT_HEADER_DIRECTORY_}")
	file(MAKE_DIRECTORY "${_OUTPUT_SOURCE_DIRECTORY_}")
	
	set(_ITR_ 0)
	foreach(X IN LISTS _LBR_${_BUNDLE_NAME_}_) # iterate over builtin resources bundle list given bundle name
		set(_CURRENT_ITEM_ "${X}")
		string(FIND "${_CURRENT_ITEM_}" "," _FOUND_ REVERSE)
		
		string(REPLACE "," ";" _ITEM_DATA_ "${_CURRENT_ITEM_}")
		list(LENGTH _ITEM_DATA_ _ITEM_D_SIZE_)
		list(GET _ITEM_DATA_ 0 _CURRENT_PATH_) # _LBR_PATH_ path for given bundle
		
		if(_ITEM_D_SIZE_ GREATER 1)
			list(SUBLIST _ITEM_DATA_ "1" "${_ITEM_D_SIZE_}" _ITEM_ALIASES_) # optional aliases for given builtin resource
		else()
			unset(_ITEM_ALIASES_)
		endif()
		
		set(_DEFAULT_ABSOLUTE_BUNDLE_SEARCH_DIRECTORY_ "${_BUNDLE_SEARCH_DIRECTORY_}")
		set(NBL_BUILTIN_RESOURCE_ABS_PATH "${_BUNDLE_SEARCH_DIRECTORY_}/${_CURRENT_PATH_}") # an absolute path to a builtin resource
		list(APPEND NBL_BUILTIN_RESOURCES "${NBL_BUILTIN_RESOURCE_ABS_PATH}")
		
		if(EXISTS "${NBL_BUILTIN_RESOURCE_ABS_PATH}")
			list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_RESOURCE_ABS_PATH}")
			file(SIZE "${NBL_BUILTIN_RESOURCE_ABS_PATH}" _FILE_SIZE_) # determine size of builtin resource in bytes
			
			macro(LIST_RESOURCE_FOR_ARCHIVER _LBR_PATH_ _LBR_FILE_SIZE_ _LBR_ID_)
				string(APPEND _RESOURCES_INIT_LIST_ "\t\t\t\t\t{\"${_LBR_PATH_}\", ${_LBR_FILE_SIZE_}, 0xdeadbeefu, ${_LBR_ID_}, nbl::system::IFileArchive::E_ALLOCATOR_TYPE::EAT_NULL},\n") # initializer list
			endmacro()
			
			LIST_RESOURCE_FOR_ARCHIVER("${_CURRENT_PATH_}" "${_FILE_SIZE_}" "${_ITR_}")
			
			foreach(_CURRENT_ALIAS_ IN LISTS _ITEM_ALIASES_)
				LIST_RESOURCE_FOR_ARCHIVER("${_CURRENT_ALIAS_}" "${_FILE_SIZE_}" "${_ITR_}")
			endforeach()
		else()
			message(FATAL_ERROR "!") # TODO: set GENERATED property, therefore we could turn some input into output and list it as builtin resource 
		endif()	
		
		math(EXPR _ITR_ "${_ITR_} + 1")
	endforeach()
	
	configure_file("${NBL_TEMPLATE_RESOURCES_ARCHIVE_HEADER}" "${_OUTPUT_HEADER_DIRECTORY_}/CArchive.h")
	
	list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_HEADER_GEN_PY}")
	list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_DATA_GEN_PY}")

	set(NBL_RESOURCES_LIST_FILE "${_OUTPUT_SOURCE_DIRECTORY_}/resources.txt")

	string(REPLACE ";" "\n" RESOURCES_ARGS "${_LBR_${_BUNDLE_NAME_}_}")
	file(WRITE "${NBL_RESOURCES_LIST_FILE}" "${RESOURCES_ARGS}")

	set(NBL_BUILTIN_RESOURCES_HEADER "${_OUTPUT_HEADER_DIRECTORY_}/${NBL_BS_HEADER_FILENAME}")
	set(NBL_BUILTIN_RESOURCE_DATA_SOURCE "${_OUTPUT_SOURCE_DIRECTORY_}/${NBL_BS_DATA_SOURCE_FILENAME}")

	add_custom_command(
		OUTPUT "${NBL_BUILTIN_RESOURCES_HEADER}" "${NBL_BUILTIN_RESOURCE_DATA_SOURCE}"
		COMMAND "${_Python3_EXECUTABLE}" "${NBL_BUILTIN_HEADER_GEN_PY}" "${NBL_BUILTIN_RESOURCES_HEADER}" "${_BUNDLE_SEARCH_DIRECTORY_}" "${NBL_RESOURCES_LIST_FILE}" "${_NAMESPACE_}" "${_GUARD_SUFFIX_}"
		COMMAND "${_Python3_EXECUTABLE}" "${NBL_BUILTIN_DATA_GEN_PY}" "${NBL_BUILTIN_RESOURCE_DATA_SOURCE}" "${_BUNDLE_SEARCH_DIRECTORY_}" "${NBL_RESOURCES_LIST_FILE}" "${_NAMESPACE_}" "${NBL_BS_HEADER_FILENAME}"
		COMMENT "Generating built-in resources"
		DEPENDS ${NBL_DEPENDENCY_FILES}
		VERBATIM  
	)
	
	add_library(${_TARGET_NAME_} STATIC
		"${NBL_BUILTIN_RESOURCES_HEADER}"
		"${NBL_BUILTIN_RESOURCE_DATA_SOURCE}"
	)
	
	target_include_directories(${_TARGET_NAME_} PUBLIC 
		"${NBL_ROOT_PATH}/include"
		"${_OUTPUT_HEADER_DIRECTORY_}"
	)
	set_target_properties(${_TARGET_NAME_} PROPERTIES CXX_STANDARD 20)
	
	if(NBL_DYNAMIC_MSVC_RUNTIME)
		set_property(TARGET ${_TARGET_NAME_} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
	else()
		set_property(TARGET ${_TARGET_NAME_} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	endif()
	
	set(NBL_BUILTIN_RESOURCES ${NBL_BUILTIN_RESOURCES}) # turn builtin resources paths list into variable
	set(NBL_BUILTIN_RESOURCES_HEADERS
		"${NBL_BUILTIN_RESOURCES_HEADER}"
		"${_OUTPUT_HEADER_DIRECTORY_}/CArchive.h"
	)
	
	macro(_ADD_PROPERTY_ _BR_PROPERTY_ _BR_PROXY_VAR_)
		get_property(_BR_PROPERTY_DEFINED_
			TARGET ${_TARGET_NAME_}
			PROPERTY ${_BR_PROPERTY_}
			DEFINED
		)

		if(NOT _BR_PROPERTY_DEFINED_)
			define_property(TARGET PROPERTY ${_BR_PROPERTY_})	
		endif()
				
		set_target_properties(${_TARGET_NAME_} PROPERTIES ${_BR_PROPERTY_} "${${_BR_PROXY_VAR_}}")
		
		unset(_BR_PROPERTY_DEFINED_)
	endmacro()
	
	_ADD_PROPERTY_(BUILTIN_RESOURCES NBL_BUILTIN_RESOURCES)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_BUNDLE_NAME _BUNDLE_NAME_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_BUNDLE_SEARCH_DIRECTORY _BUNDLE_SEARCH_DIRECTORY_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_NAMESPACE _NAMESPACE_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_HEADER_DIRECTORY _OUTPUT_HEADER_DIRECTORY_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_SOURCE_DIRECTORY _OUTPUT_SOURCE_DIRECTORY_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_HEADERS NBL_BUILTIN_RESOURCES_HEADERS)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY _OUTPUT_INCLUDE_SEARCH_DIRECTORY_)
endfunction()