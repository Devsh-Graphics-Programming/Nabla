# Assigns builtin resources to a bundle a target library will be created with
# _BUNDLE_NAME_ is a bundle name, must be a valid CMake list variable
# _LBR_PATH_ is a path to builtin resource
# optional aliases may be preset after the _LBR_PATH_

macro(LIST_BUILTIN_RESOURCE _BUNDLE_NAME_ _LBR_PATH_)
	math(EXPR _ALIAS_C_ "${ARGC} - 2" OUTPUT_FORMAT DECIMAL)
	set(_ALIAS_ARGS_ ${ARGV})
	
	if("${_ALIAS_C_}" GREATER "0")
		list(SUBLIST _ALIAS_ARGS_ "2" "${_ALIAS_C_}" _ALIAS_ARGS_)
		
		foreach(_ALIAS_ IN LISTS _ALIAS_ARGS_)
			string(APPEND _OPTIONAL_ALIASES_ ",${_ALIAS_}")
		endforeach()
	endif()
	
	list(FIND ${_BUNDLE_NAME_} "${_LBR_PATH_}" _NBL_FOUND_)
	
	if(NOT "${_NBL_FOUND_}" STREQUAL "-1")
		message(FATAL_ERROR "Duplicated \"${_LBR_PATH_}\" builtin resource list-request detected to \"${_BUNDLE_NAME_}\", remove the entry!")
	endif()
	
	list(APPEND ${_BUNDLE_NAME_} "${_LBR_PATH_}")
	set(${_BUNDLE_NAME_} ${${_BUNDLE_NAME_}}) # override
	
	list(APPEND _LBR_${_BUNDLE_NAME_}_ "${_LBR_PATH_}${_OPTIONAL_ALIASES_}")
	set(_LBR_${_BUNDLE_NAME_}_ ${_LBR_${_BUNDLE_NAME_}_}) # override
	
	unset(_OPTIONAL_ALIASES_)
	unset(_ALIAS_ARGS_)
endmacro()

# Creates a library with builtin resources for given bundle
# _TARGET_NAME_ is name of a target library that will be created
# _BUNDLE_NAME_ a list variable populated using LIST_BUILTIN_RESOURCE
# _BUNDLE_SEARCH_DIRECTORY_ is an absolute search directory path for builtin resorces for given bundle
# _BUNDLE_ARCHIVE_ABSOLUTE_PATH_ is a "absolute path" for an archive which will store a given bundle of builtin resources, must be relative _BUNDLE_SEARCH_DIRECTORY_
# _NAMESPACE_ is a C++ namespace builtin resources will be wrapped into
# _OUTPUT_INCLUDE_SEARCH_DIRECTORY_ is an absolute path to output directory for builtin resources header files which will be a search directory for generated headers outputed to ${_OUTPUT_HEADER_DIRECTORY_}/${_NAMESPACE_PREFIX_} where namespace prefix is the namespace turned into a path
# _OUTPUT_SOURCE_DIRECTORY_ is an absolute path to output directory for builtin resources source files
# _STATIC_ optional last argument is a bool, if true then add_library will use STATIC, SHARED otherwise. Pay attention that MSVC runtime is controlled by NBL_COMPILER_DYNAMIC_RUNTIME which is not an argument of this function
#
# As an example one could list a resource as following
# LIST_BUILTIN_RESOURCE(SOME_RESOURCES_TO_EMBED "glsl/blit/default_compute_normalization.comp")
# and then create builtin resource target with the resource above using
# ADD_CUSTOM_BUILTIN_RESOURCES("aTarget" SOME_RESOURCES_TO_EMBED "${NBL_ROOT_PATH}/include" "nbl/builtin" "myns::builtin" "${NBL_ROOT_PATH_BINARY}/include" "${NBL_ROOT_PATH_BINARY}/src")
# a real absolute path to the resource on the disk would be ${NBL_ROOT_PATH}/include/nbl/builtin/glsl/blit/default_compute_normalization.comp
# the builtin resource path seen in Nabla filesystem would be "nbl/builtin/builtin/glsl/blit/default_compute_normalization.comp" where "nbl/builtin" would be an absolute path for an archive

function(ADD_CUSTOM_BUILTIN_RESOURCES _TARGET_NAME_ _BUNDLE_NAME_ _BUNDLE_SEARCH_DIRECTORY_ _BUNDLE_ARCHIVE_ABSOLUTE_PATH_ _NAMESPACE_ _OUTPUT_INCLUDE_SEARCH_DIRECTORY_ _OUTPUT_SOURCE_DIRECTORY_)
	if(NOT DEFINED _Python3_EXECUTABLE)
		message(FATAL_ERROR "_Python3_EXECUTABLE must be defined - call find_package(Python3 COMPONENTS Interpreter REQUIRED)")
	endif()
	
	if("${ARGV7}" STREQUAL "SHARED")
		set(_LIB_TYPE_ SHARED)
		set(_SHARED_ True)
		set(NBL_BR_API "NBL_BR_API")
	else()
		set(_LIB_TYPE_ STATIC)
		set(_SHARED_ False)
		unset(NBL_BR_API)
	endif()
	
	if("${ARGV8}" STREQUAL "INTERNAL")
		set(_NBL_INTERNAL_BR_CREATION_ ON)
	else()
		set(_NBL_INTERNAL_BR_CREATION_ OFF)
	endif()

	set(NBL_TEMPLATE_RESOURCES_ARCHIVE_HEADER "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/template/CArchive.h.in")
	set(NBL_BUILTIN_HEADER_GEN_PY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/builtinHeaderGen.py")
	set(NBL_BUILTIN_DATA_GEN_PY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/builtinDataGen.py")
	set(NBL_BS_HEADER_FILENAME "builtinResources.h")
	set(NBL_BS_DATA_SOURCE_FILENAME "builtinResourceData.cpp")
	
	string(REPLACE "::" "/" _NAMESPACE_PREFIX_ "${_NAMESPACE_}")
	string(REPLACE "::" "_" _GUARD_SUFFIX_ "${_NAMESPACE_}")
	string(REGEX REPLACE "^[0-9]+\." "" _GUARD_SUFFIX_ ${_GUARD_SUFFIX_})
	string(TOUPPER ${_GUARD_SUFFIX_} _GUARD_SUFFIX_)
	string(MAKE_C_IDENTIFIER ${_GUARD_SUFFIX_} _GUARD_SUFFIX_)
	
	set(_OUTPUT_INCLUDE_SEARCH_DIRECTORY_ "${_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}")
	set(_OUTPUT_HEADER_DIRECTORY_ "${_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}/${_NAMESPACE_PREFIX_}")
	
	file(MAKE_DIRECTORY "${_OUTPUT_HEADER_DIRECTORY_}")
	file(MAKE_DIRECTORY "${_OUTPUT_SOURCE_DIRECTORY_}")
	
	set(_ITR_ 0)
	foreach(X IN LISTS _LBR_${_BUNDLE_NAME_}_) # iterate over builtin resources bundle list given bundle name
		set(_CURRENT_ITEM_ "${X}")
		
		string(REPLACE "," ";" _ITEM_DATA_ "${_CURRENT_ITEM_}")
		list(LENGTH _ITEM_DATA_ _ITEM_D_SIZE_)
		list(GET _ITEM_DATA_ 0 _CURRENT_PATH_) # _LBR_PATH_ path for given bundle
		
		set(_BUNDLE_ARCHIVE_ABSOLUTE_PATH_ "${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}")
		set(NBL_BUILTIN_RESOURCE_ABS_PATH "${_BUNDLE_SEARCH_DIRECTORY_}/${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}/${_CURRENT_PATH_}") # an absolute path to a resource a builtin resource will be created as
		
		list(APPEND NBL_BUILTIN_RESOURCES "${NBL_BUILTIN_RESOURCE_ABS_PATH}")
		list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_RESOURCE_ABS_PATH}")
		
		# validate
		if(NOT EXISTS "${NBL_BUILTIN_RESOURCE_ABS_PATH}")
			get_source_file_property(NBL_BUILTIN_IS_GENERATED "${NBL_BUILTIN_RESOURCE_ABS_PATH}" GENERATED)
			
			if(NOT NBL_BUILTIN_IS_GENERATED)
				message(FATAL_ERROR "You have requested '${NBL_BUILTIN_RESOURCE_ABS_PATH}' to be builtin resource but it doesn't exist or is not marked with GENERATED property!")
			endif()
		endif()	
		
		math(EXPR _ITR_ "${_ITR_} + 1")
	endforeach()
	
	set(NBL_BUILTIN_DATA_ARCHIVE_H "${_OUTPUT_HEADER_DIRECTORY_}/CArchive.h")
	set(NBL_BUILTIN_DATA_ARCHIVE_CPP "${_OUTPUT_SOURCE_DIRECTORY_}/CArchive.cpp")
	
	list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_HEADER_GEN_PY}")
	list(APPEND NBL_DEPENDENCY_FILES "${NBL_BUILTIN_DATA_GEN_PY}")

	set(NBL_RESOURCES_LIST_FILE "${_OUTPUT_SOURCE_DIRECTORY_}/resources.txt")

	string(REPLACE ";" "\n" RESOURCES_ARGS "${_LBR_${_BUNDLE_NAME_}_}")
	file(WRITE "${NBL_RESOURCES_LIST_FILE}" "${RESOURCES_ARGS}")

	set(NBL_BUILTIN_RESOURCES_H "${_OUTPUT_HEADER_DIRECTORY_}/${NBL_BS_HEADER_FILENAME}")
	set(NBL_BUILTIN_RESOURCE_DATA_CPP "${_OUTPUT_SOURCE_DIRECTORY_}/${NBL_BS_DATA_SOURCE_FILENAME}")
	
	if(NBL_BR_FORCE_CONSTEXPR_HASH)
		set(_NBL_BR_RUNTIME_HASH_ 0)
	else()
		set(_NBL_BR_RUNTIME_HASH_ 1)
	endif()

	set(NBL_BUILTIN_RESOURCES_COMMON_ARGS
   		--resourcesFile "${NBL_RESOURCES_LIST_FILE}"
   		--resourcesNamespace "${_NAMESPACE_}"
	)

	add_custom_command(OUTPUT "${NBL_BUILTIN_RESOURCES_H}" "${NBL_BUILTIN_RESOURCE_DATA_CPP}" "${NBL_BUILTIN_DATA_ARCHIVE_H}" "${NBL_BUILTIN_DATA_ARCHIVE_CPP}"
		COMMAND "${_Python3_EXECUTABLE}" "${NBL_BUILTIN_HEADER_GEN_PY}" ${NBL_BUILTIN_RESOURCES_COMMON_ARGS} --outputBuiltinPath "${NBL_BUILTIN_RESOURCES_H}" --outputArchivePath "${NBL_BUILTIN_DATA_ARCHIVE_H}" --archiveBundlePath "${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}" --guardSuffix "${_GUARD_SUFFIX_}" --isSharedLibrary "${_SHARED_}"
		COMMAND "${_Python3_EXECUTABLE}" "${NBL_BUILTIN_DATA_GEN_PY}" ${NBL_BUILTIN_RESOURCES_COMMON_ARGS} --outputBuiltinPath "${NBL_BUILTIN_RESOURCE_DATA_CPP}" --outputArchivePath "${NBL_BUILTIN_DATA_ARCHIVE_CPP}" --bundleAbsoluteEntryPath "${_BUNDLE_SEARCH_DIRECTORY_}/${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}" --correspondingHeaderFile "${NBL_BS_HEADER_FILENAME}" --xxHash256Exe "$<${_NBL_BR_RUNTIME_HASH_}:$<TARGET_FILE:xxHash256>>"
		COMMENT "Generating \"${_TARGET_NAME_}\"'s sources & headers"
		DEPENDS ${NBL_DEPENDENCY_FILES}
		VERBATIM
	)
	
	add_library(${_TARGET_NAME_} ${_LIB_TYPE_}
		"${NBL_BUILTIN_RESOURCES_H}"
		"${NBL_BUILTIN_RESOURCE_DATA_CPP}"
		"${NBL_BUILTIN_DATA_ARCHIVE_H}"
		"${NBL_BUILTIN_DATA_ARCHIVE_CPP}"
	)
	
	if(NBL_FORCE_RELEASE_3RDPARTY) # priority over RWDI
		nbl_adjust_flags(TARGET ${_TARGET_NAME_} MAP_RELEASE Release MAP_RELWITHDEBINFO Release MAP_DEBUG Release)
	elseif(NBL_FORCE_RELWITHDEBINFO_3RDPARTY)
		nbl_adjust_flags(TARGET ${_TARGET_NAME_} MAP_RELEASE RelWithDebInfo MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG RelWithDebInfo)
	else()
		nbl_adjust_flags(TARGET ${_TARGET_NAME_} MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
	endif()
	
	set(_BR_CONSTEXPR_STEPS_ 696969696969)

	if(MSVC)
		list(APPEND _BR_COMPILE_OPTIONS_ /constexpr:steps${_BR_CONSTEXPR_STEPS_})
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		list(APPEND _BR_COMPILE_OPTIONS_ -fconstexpr-ops-limit=${_BR_CONSTEXPR_STEPS_})
	else()
		list(APPEND _BR_COMPILE_OPTIONS_ -fconstexpr-steps=${_BR_CONSTEXPR_STEPS_})
	endif()
	
	target_compile_options(${_TARGET_NAME_} PRIVATE
		${_BR_COMPILE_OPTIONS_}
	)
	
	set_target_properties(${_TARGET_NAME_} PROPERTIES
        DISABLE_PRECOMPILE_HEADERS ON
    )
	
	if(_LIB_TYPE_ STREQUAL SHARED)
		target_compile_definitions(${_TARGET_NAME_} 
			PRIVATE __NBL_BUILDING_TARGET__
		)
	endif()
	
	if(TARGET Nabla)
		get_target_property(_NABLA_INCLUDE_DIRECTORIES_ Nabla INCLUDE_DIRECTORIES)
		
		if(NBL_STATIC_BUILD AND _LIB_TYPE_ STREQUAL SHARED)
			message(FATAL_ERROR "Nabla must be built as dynamic library in order to combine this tool with SHARED setup!")
		endif()
		
		if(NOT _NBL_INTERNAL_BR_CREATION_)
			target_link_libraries(${_TARGET_NAME_} PUBLIC Nabla)
		endif()
	endif()
	
	if(NOT DEFINED _NABLA_INCLUDE_DIRECTORIES_) # TODO, validate by populating generator expressions if any and checking whether a path to the BuildConfigOptions.h exists per config
		if(NOT _NBL_INTERNAL_BR_CREATION_) # trust internal Nabla BR targets, include search paths may be added later
			message(FATAL_ERROR "_NABLA_INCLUDE_DIRECTORIES_ has been not found. You are required to define _NABLA_INCLUDE_DIRECTORIES_ containing at least include search directory path to BuildConfigOptions.h")
		endif()
	endif()
	
	target_include_directories(${_TARGET_NAME_} PUBLIC 
		"${_NABLA_INCLUDE_DIRECTORIES_}"
		"${_OUTPUT_HEADER_DIRECTORY_}"
	)
	set_target_properties(${_TARGET_NAME_} PROPERTIES CXX_STANDARD 20)

	set_property(TARGET ${_TARGET_NAME_} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>$<$<BOOL:${NBL_COMPILER_DYNAMIC_RUNTIME}>:DLL>")
	
	set(NBL_BUILTIN_RESOURCES ${NBL_BUILTIN_RESOURCES}) # turn builtin resources paths list into variable
	
	set(NBL_BUILTIN_RESOURCES_HEADERS
		"${NBL_BUILTIN_RESOURCES_H}"
		"${NBL_BUILTIN_DATA_ARCHIVE_H}"
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
	_ADD_PROPERTY_(BUILTIN_RESOURCES_BUNDLE_ARCHIVE_ABSOLUTE_PATH _BUNDLE_ARCHIVE_ABSOLUTE_PATH_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_NAMESPACE _NAMESPACE_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_HEADER_DIRECTORY _OUTPUT_HEADER_DIRECTORY_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_SOURCE_DIRECTORY _OUTPUT_SOURCE_DIRECTORY_)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_HEADERS NBL_BUILTIN_RESOURCES_HEADERS)
	_ADD_PROPERTY_(BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY _OUTPUT_INCLUDE_SEARCH_DIRECTORY_)
	
	if(MSVC AND NBL_SANITIZE_ADDRESS)
		set_property(TARGET ${_TARGET_NAME_} PROPERTY COMPILE_OPTIONS /fsanitize=address)
	endif()
endfunction()

function(NBL_REGISTER_SPIRV_SHADERS)
	cmake_parse_arguments(IMPL "" "DISCARD;LINK_TO;MOUNT_POINT_DEFINE" "PERMUTE;REQUIRED;ARCHIVE;INPUTS" ${ARGN})

	if(NOT IMPL_MOUNT_POINT_DEFINE)
		message(FATAL_ERROR "MOUNT_POINT_DEFINE argument missing!")
	endif()

	if(NOT IMPL_ARCHIVE)
		message(FATAL_ERROR "ARCHIVE arguments missing!")
	endif()

	cmake_parse_arguments(IMPL "" "TARGET;INPUT_DIRECTORY;NAMESPACE;PREFIX" "" ${IMPL_ARCHIVE})

	if(NOT IMPL_TARGET)
		message(FATAL_ERROR "Missing TARGET argument in ARCHIVE specification!")
	endif()

	if(NOT IMPL_INPUT_DIRECTORY)
		message(FATAL_ERROR "Missing INPUT_DIRECTORY argument in ARCHIVE specification!")
	endif()

	if(NOT IMPL_NAMESPACE)
		message(FATAL_ERROR "Missing NAMESPACE argument in ARCHIVE specification!")
	endif()

	set(_BUNDLE_ARCHIVE_ABSOLUTE_PATH_ ${IMPL_PREFIX})
	get_filename_component(_BUNDLE_SEARCH_DIRECTORY_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/spirv/shaders/mount-point" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/spirv/archive/src" ABSOLUTE)
	get_filename_component(_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/spirv/archive/include" ABSOLUTE)

	set(_BUILTIN_RESOURCES_NAMESPACE_ ${IMPL_NAMESPACE})
	set(_LINK_MODE_ STATIC)

	get_filename_component(BUILTIN_ARCHIVE_INPUT_ABS_ENTRY "${IMPL_INPUT_DIRECTORY}" ABSOLUTE)
	set(BUILTIN_KEY_ENTRY_ABS "${BUILTIN_ARCHIVE_INPUT_ABS_ENTRY}/${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}")
	
	# REMOVE IT AND ADD "DEPENDS" to INPUTS
	# file(GLOB_RECURSE _DEPENDS_ON_ CONFIGURE_DEPENDS "${BUILTIN_KEY_ENTRY_ABS}/*.hlsl")
	# list(FILTER _DEPENDS_ON_ EXCLUDE REGEX /preprocessed.hlsl)
	# and maybe extra DEPENDS shared for all inputs
	####################

	set(SPIRV_OUTPUT_ARCHIVE_KEY_ABS_ENTRY_DIR "${_BUNDLE_SEARCH_DIRECTORY_}/${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}")

	set(DEVICE_CONFIG_ENTRY_DIR "${_BUNDLE_SEARCH_DIRECTORY_}/PermutationCaps")
	set(DEVICE_CONFIG_TEMPLATE_PATH "${DEVICE_CONFIG_ENTRY_DIR}/DeviceConfig.hlsl")
	file(REMOVE_RECURSE "${DEVICE_CONFIG_ENTRY_DIR}/")

	set(DEVICE_CONFIG_VIEW
[=[

// -> this code has been autogenerated!
#ifndef _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
#define _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
#ifdef __HLSL_VERSION
#include <nbl/builtin/hlsl/cpp_compat/basic.h>
struct DeviceConfigCaps
{
@CAPS_EVAL@
};
#endif // __HLSL_VERSION
#endif // _PERMUTATION_CAPS_AUTO_GEN_GLOBALS_INCLUDED_
// <- this code has been autogenerated!

// we inject our own config above
#define NBL_USE_SPIRV_BUILTINS

/*
note: (**)
we have a bug and I cannot use -D to create
define with dxc options, it gets ignored, so
temporary I will create int input files and 
inject this config code

#ifndef NBL_DYMANIC_INCLUDE
#error "NBL_DYMANIC_INCLUDE must be defined!"
#endif // NBL_DYMANIC_INCLUDE

// proxy HLSL input with #define
#include NBL_DYMANIC_INCLUDE
*/
]=]
)
	set(KEY_EXTENSION .spv)

	if(IMPL_PERMUTE)
	    list(LENGTH IMPL_PERMUTE KEYS_LENGTH)
		math(EXPR TOTAL_INDEX_RANGE "(1 << ${KEYS_LENGTH}) - 1")
	else()
		set(TOTAL_INDEX_RANGE 0)
	endif()

    foreach(INDEX RANGE 0 ${TOTAL_INDEX_RANGE})
        set(BIT_INDEX 0)
		unset(CAPS_EVAL)
		unset(POSTFIX_ACCESS_KEY)

        foreach(KEY IN LISTS IMPL_PERMUTE)
            math(EXPR BIT "((${INDEX} >> ${BIT_INDEX}) & 1)")
            if(BIT EQUAL 1)
                set(STATE "true")
            else()
                set(STATE "false")
            endif()
			string(APPEND POSTFIX_ACCESS_KEY "_${KEY}=${STATE}")
            string(APPEND CAPS_EVAL "NBL_CONSTEXPR_STATIC_INLINE bool ${KEY} = ${STATE}; // got permuted\n")
            math(EXPR BIT_INDEX "${BIT_INDEX} + 1")
        endforeach()

		foreach(KEY IN LISTS IMPL_REQUIRED)
			string(APPEND CAPS_EVAL "NBL_CONSTEXPR_STATIC_INLINE bool ${KEY} = true; // always required\n")
		endforeach()

		# generate permuted config
		set(PERMUTED_DEVICE_CONFIG "${DEVICE_CONFIG_TEMPLATE_PATH}${POSTFIX_ACCESS_KEY}")
		list(APPEND DEVICE_CONFIG_FILES "${PERMUTED_DEVICE_CONFIG}")
		string(CONFIGURE "${DEVICE_CONFIG_VIEW}" CONFIG_CONTENT @ONLY)
		file(WRITE "${PERMUTED_DEVICE_CONFIG}" "${CONFIG_CONTENT}")

		# create compile rules for given input with permuted config
		set(i 0)
		list(LENGTH IMPL_INPUTS LEN)
		while(i LESS LEN)
			list(GET IMPL_INPUTS ${i} TOKEN)
			if(TOKEN STREQUAL "KEY")
				math(EXPR i "${i} + 1")
				list(GET IMPL_INPUTS ${i} FILEPATH)
				set(COMPILE_OPTIONS "")
				math(EXPR i "${i} + 1")
	
				list(GET IMPL_INPUTS ${i} NEXT)
				if(NOT NEXT STREQUAL "COMPILE_OPTIONS")
					message(FATAL_ERROR "Expected COMPILE_OPTIONS after KEY ${FILEPATH}")
				endif()
				math(EXPR i "${i} + 1")
	
				while(i LESS LEN)
					list(GET IMPL_INPUTS ${i} ARG)
					if(ARG STREQUAL "KEY")
						break()
					endif()
					list(APPEND COMPILE_OPTIONS "${ARG}")
					math(EXPR i "${i} + 1")
				endwhile()

				set(IMPL_KEY ${FILEPATH})
				set(TARGET_KEY "${IMPL_KEY}${POSTFIX_ACCESS_KEY}${KEY_EXTENSION}")

				if(IMPL_DISCARD AND "${POSTFIX_ACCESS_KEY}" MATCHES "${IMPL_DISCARD}")
					if(NBL_LOG_VERBOSE)
						message(STATUS "[Nabla Builtin SPIRV]: Discarded \"${TARGET_KEY}\" key for ${IMPL_TARGET}")
					endif()
					continue()
				endif()

				if(NBL_LOG_VERBOSE)
					message(STATUS "[Nabla Builtin SPIRV]: Registered \"${TARGET_KEY}\" key ${IMPL_TARGET}")
				endif()

				set(TARGET_INPUT "${BUILTIN_KEY_ENTRY_ABS}/${IMPL_KEY}")
				list(APPEND REQUESTED_INPUTS "${TARGET_INPUT}")
				set(TAGET_OUTPUT "${SPIRV_OUTPUT_ARCHIVE_KEY_ABS_ENTRY_DIR}/${TARGET_KEY}")
				list(APPEND SPIRV_OUTPUTS "${TAGET_OUTPUT}")

				# doing as workaround for (**), dynamic define include could be better because then I don't have to generate intermediate files with glue at configure time
				set(INT_INPUT "${SPIRV_OUTPUT_ARCHIVE_KEY_ABS_ENTRY_DIR}/.int/${TARGET_KEY}")
				list(APPEND INT_FILES "${INT_INPUT}")
				set(INT_INPUT_VIEW
[=[

@INPUT_CONFIG_CONTENT@

#include "@PERMUTED_DEVICE_CONFIG@"
#include "@TARGET_INPUT@"

]=]		
				)

				string(CONFIGURE "${INT_INPUT_VIEW}" INT_CONTENT @ONLY)
				file(WRITE "${INT_INPUT}" "${INT_CONTENT}")

				# would get added by NSC anyway and spam in output
				set(REQUIRED_OPTIONS
					-HV 202x 
					-Wno-c++14-extensions 
					-Wno-gnu-static-float-init 
					-Wno-c++1z-extensions 
					-Wno-c++11-extensions 
					-fvk-use-scalar-layout 
					-enable-16bit-types 
					-Zpr 
					-spirv 
					-fspv-target-env=vulkan1.3
				)

				set(NBL_NSC_COMPILE_COMMAND
					"$<TARGET_FILE:nsc>"
					-Fc "${TAGET_OUTPUT}"
					${COMPILE_OPTIONS} ${REQUIRED_OPTIONS}
					# "-DNBL_DYMANIC_INCLUDE=<${TARGET_INPUT}>" # (**)!
					"${INT_INPUT}"
				)

				add_custom_command(OUTPUT "${TAGET_OUTPUT}"
					COMMAND ${NBL_NSC_COMPILE_COMMAND}
					DEPENDS ${_DEPENDS_ON_} ${INT_INPUT}
					COMMENT "Creating ${TAGET_OUTPUT}"
					VERBATIM
					COMMAND_EXPAND_LISTS
				)

				LIST_BUILTIN_RESOURCE(NBL_RESOURCES_TO_EMBED ${TARGET_KEY})
			else()
				math(EXPR i "${i} + 1")
			endif()
		endwhile()
    endforeach()

	if(NBL_EMBED_BUILTIN_RESOURCES)
		ADD_CUSTOM_BUILTIN_RESOURCES(${IMPL_TARGET} NBL_RESOURCES_TO_EMBED "${_BUNDLE_SEARCH_DIRECTORY_}" "${_BUNDLE_ARCHIVE_ABSOLUTE_PATH_}" "${_BUILTIN_RESOURCES_NAMESPACE_}" "${_OUTPUT_DIRECTORY_HEADER_}" "${_OUTPUT_DIRECTORY_SOURCE_}" "${_LINK_MODE_}")
	else()
		add_library(${IMPL_TARGET} INTERFACE)
	endif()

	target_compile_definitions(${IMPL_TARGET} INTERFACE ${IMPL_MOUNT_POINT_DEFINE}="${_BUNDLE_SEARCH_DIRECTORY_}")

	if(IMPL_LINK_TO)
		if(NBL_EMBED_BUILTIN_RESOURCES)
			LINK_BUILTIN_RESOURCES_TO_TARGET(${IMPL_LINK_TO} ${IMPL_TARGET})
		else()
			target_link_libraries(${IMPL_LINK_TO} INTERFACE ${IMPL_TARGET})
		endif()
	endif()

	set(HEADER_ONLY ${INT_FILES} ${DEVICE_CONFIG_FILES} ${REQUESTED_INPUTS} ${SPIRV_OUTPUTS})
	target_sources(${IMPL_TARGET} PRIVATE ${HEADER_ONLY})
	set_source_files_properties(${HEADER_ONLY} PROPERTIES HEADER_FILE_ONLY TRUE)

	set(RTE "Resources to embed")
	set(IN "${RTE}/In")
	set(OUT "${RTE}/Out")

	source_group("${IN}/Intermediate" FILES ${INT_FILES})
	source_group("${IN}/Device Configs" FILES ${DEVICE_CONFIG_FILES})
	source_group("${IN}" FILES ${REQUESTED_INPUTS})
	source_group("${OUT}" FILES ${SPIRV_OUTPUTS})
endfunction()