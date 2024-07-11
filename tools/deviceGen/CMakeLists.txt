get_filename_component(SRC_VIDEO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/input" ABSOLUTE)
get_filename_component(NBL_DEVICE_GEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE) # TODO: fix a bug in checking for GENERATED property which doesnt eveluate properly generator expressions so $<CONFIG> fails

set(NBL_DEVICE_GEN_INCLUDE_DIR "${NBL_DEVICE_GEN_INCLUDE_DIR}" CACHE INTERNAL "" FORCE)
set(NBL_DEVICE_GEN_OUTPUT_DIR "${NBL_DEVICE_GEN_INCLUDE_DIR}/nbl/video")

# inputs
set(INPUT_DL_JSON "${SRC_VIDEO_PATH}/device_limits.json")
set(INPUT_DF_JSON "${SRC_VIDEO_PATH}/device_features.json")

set(INPUT_JSONS
  "${INPUT_DL_JSON}"
  "${INPUT_DF_JSON}"
)

# outputs
set(OUTPUT_DL_TEMPLATE "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceLimits_")
set(OUTPUT_DF_TEMPLATE "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceFeatures_")
set(OUTPUT_DT_TEMPLATE "${NBL_DEVICE_GEN_OUTPUT_DIR}/device_capabilities_traits_")

set(OUTPUT_DL_MEMBERS "${OUTPUT_DL_TEMPLATE}members.h")
set(OUTPUT_DL_SUBSET "${OUTPUT_DL_TEMPLATE}subset.h")
set(OUTPUT_DF_MEMBERS "${OUTPUT_DF_TEMPLATE}members.h")
set(OUTPUT_DF_UNION "${OUTPUT_DF_TEMPLATE}union.h")
set(OUTPUT_DF_INTERSECT "${OUTPUT_DF_TEMPLATE}intersect.h")
set(OUTPUT_DT_MEMBERS "${OUTPUT_DT_TEMPLATE}members.hlsl")
set(OUTPUT_DT_TESTERS "${OUTPUT_DT_TEMPLATE}testers.hlsl")
set(OUTPUT_DT_DEFAULTS "${OUTPUT_DT_TEMPLATE}defaults.hlsl")
set(OUTPUT_DT_FLOATS "${OUTPUT_DT_TEMPLATE}floats.hlsl")
set(OUTPUT_DT_ENUMS "${OUTPUT_DT_TEMPLATE}enums.hlsl")
set(OUTPUT_DT_JIT "${OUTPUT_DT_TEMPLATE}jit.h")

set(OUTPUT_HEADERS 
"${OUTPUT_DL_MEMBERS}"
"${OUTPUT_DL_SUBSET}"
"${OUTPUT_DF_MEMBERS}"
"${OUTPUT_DF_UNION}"
"${OUTPUT_DF_INTERSECT}"
"${OUTPUT_DT_MEMBERS}"
"${OUTPUT_DT_TESTERS}"
"${OUTPUT_DT_DEFAULTS}"
"${OUTPUT_DT_FLOATS}"
"${OUTPUT_DT_ENUMS}"
"${OUTPUT_DT_JIT}"
)

# python script
set(NBL_GEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/gen.py")
set(NBL_DGEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/DeviceGen.py")

# custom command
set(NBL_COMMAND
  "${_Python3_EXECUTABLE}"
  "${NBL_GEN_PY}"
  "--limits_json_path" "${INPUT_DL_JSON}"  
  "--features_json_path" "${INPUT_DF_JSON}"
  "--limits_output_members_path" "${OUTPUT_DL_MEMBERS}"
  "--limits_output_subset_path" "${OUTPUT_DL_SUBSET}"
  "--features_output_members_path" "${OUTPUT_DF_MEMBERS}"
  "--features_output_union_path" "${OUTPUT_DF_UNION}"
  "--features_output_intersect_path" "${OUTPUT_DF_INTERSECT}"
  "--traits_output_members_path" "${OUTPUT_DT_MEMBERS}"
  "--traits_output_testers_path" "${OUTPUT_DT_TESTERS}"
  "--traits_output_defaults_path" "${OUTPUT_DT_DEFAULTS}"
  "--traits_output_enums_path" "${OUTPUT_DT_ENUMS}"
  "--traits_output_floats_path" "${OUTPUT_DT_FLOATS}"
  "--jit_traits_output_path" "${OUTPUT_DT_JIT}"
)

set(NBL_DEPENDS
  ${INPUT_JSONS}
  "${NBL_GEN_PY}"
  "${NBL_DGEN_PY}"
)

add_custom_command(OUTPUT ${OUTPUT_HEADERS}
    DEPENDS ${NBL_DEPENDS}
    COMMAND ${NBL_COMMAND}
    VERBATIM
)

add_custom_target(DeviceHeaders DEPENDS ${OUTPUT_HEADERS})

LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/device_capabilities_traits_members.hlsl")
LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/device_capabilities_traits_testers.hlsl")
LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/device_capabilities_traits_defaults.hlsl")
LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/device_capabilities_traits_floats.hlsl")
LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/device_capabilities_traits_enums.hlsl")

get_filename_component(_DEVICE_GEN_BR_OUTPUT_DIRECTORY_HEADER_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/include" ABSOLUTE)
get_filename_component(_DEVICE_GEN_BR_OUTPUT_DIRECTORY_SOURCE_ "${CMAKE_CURRENT_BINARY_DIR}/builtin/src" ABSOLUTE)

ADD_CUSTOM_BUILTIN_RESOURCES(deviceGenBuiltinResourceData NBL_DEVICE_GEN_RESOURCES_TO_EMBED "${NBL_DEVICE_GEN_INCLUDE_DIR}" "nbl" "nbl::devicegen::builtin" "${_DEVICE_GEN_BR_OUTPUT_DIRECTORY_HEADER_}" "${_DEVICE_GEN_BR_OUTPUT_DIRECTORY_SOURCE_}" "STATIC" "INTERNAL")
add_dependencies(deviceGenBuiltinResourceData DeviceHeaders)

if(NBL_EMBED_BUILTIN_RESOURCES)
get_target_property(_DEVICE_GEN_BR_OUTPUT_INCLUDE_SEARCH_DIRECTORY_ deviceGenBuiltinResourceData BUILTIN_RESOURCES_INCLUDE_SEARCH_DIRECTORY)
target_include_directories(deviceGenBuiltinResourceData PUBLIC "${_NABLA_INCLUDE_DIRECTORIES_}") # the reason behind the patch is because this internal BR target gets created before Nabla
target_include_directories(Nabla PUBLIC 
"${_DEVICE_GEN_BR_OUTPUT_INCLUDE_SEARCH_DIRECTORY_}"
)
add_dependencies(Nabla deviceGenBuiltinResourceData)
if(NBL_STATIC_BUILD)
		set(_NBL_LINK_QUALIFIER_ INTERFACE)
	else()
		set(_NBL_LINK_QUALIFIER_ PRIVATE)
	endif()
target_link_libraries(Nabla ${_NBL_LINK_QUALIFIER_}
		deviceGenBuiltinResourceData
	)
endif()

add_dependencies(Nabla DeviceHeaders)
target_include_directories(Nabla PUBLIC "${NBL_DEVICE_GEN_INCLUDE_DIR}")