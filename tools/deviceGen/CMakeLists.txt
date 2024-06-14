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

# outputs, why they don't have extensions like .h?
# they are being used in multiple headers with the same output, so the extensions are added by the script
# now we have ap roblem because the custom command should known there is an output with .h file, but from its perspective there is not, w8

# TODO: remove _, use full paths with .h ext
# TMP fix bellow

set(OUTPUT_DL_HEADER "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceLimits_")
set(OUTPUT_DF_HEADER "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceFeatures_")
set(OUTPUT_DT_HEADER "${NBL_DEVICE_GEN_OUTPUT_DIR}/device_capabilities_traits_")

set(OUTPUT_DL_HEADER2 "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceLimits.h")
set(OUTPUT_DF_HEADER2 "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceFeatures.h")
set(OUTPUT_DT_HEADER2 "${NBL_DEVICE_GEN_OUTPUT_DIR}/device_capabilities_traits.h")

# this needs to have paths to REAL locations not with "_" for the script gen purpose
set(OUTPUT_HEADERS2 
  "${OUTPUT_DL_HEADER2}"
  "${OUTPUT_DF_HEADER2}"
  "${OUTPUT_DT_HEADER2}"
)

# python script
set(NBL_GEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/gen.py")

# custom command
set(NBL_COMMAND
  "${_Python3_EXECUTABLE}"
  "${NBL_GEN_PY}"
  "${INPUT_DL_JSON}"
  "${INPUT_DF_JSON}"
  "${OUTPUT_DL_HEADER}"
  "${OUTPUT_DF_HEADER}"
  "${OUTPUT_DT_HEADER}"
)

add_custom_command(OUTPUT ${OUTPUT_HEADERS2}
    DEPENDS ${INPUT_JSONS}
    COMMAND ${NBL_COMMAND}
)

add_custom_target(DeviceHeaders DEPENDS ${OUTPUT_HEADERS2})
set(NBL_DEVICE_GEN_HEADERS ${OUTPUT_HEADERS2})

message(WARNING "OUTPUT_HEADERS2: ${OUTPUT_HEADERS2}")

LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/SPhysicalDeviceLimits.h")
LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/SPhysicalDeviceFeatures.h")
LIST_BUILTIN_RESOURCE(NBL_DEVICE_GEN_RESOURCES_TO_EMBED "video/device_capabilities_traits.h")

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