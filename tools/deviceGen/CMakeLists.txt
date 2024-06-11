get_filename_component(SRC_VIDEO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/input" ABSOLUTE)
get_filename_component(NBL_DEVICE_GEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/include" ABSOLUTE)

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
set(OUTPUT_DL_HEADER "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceLimits_")
set(OUTPUT_DF_HEADER "${NBL_DEVICE_GEN_OUTPUT_DIR}/SPhysicalDeviceFeatures_")

set(OUTPUT_HEADERS 
  "${OUTPUT_DL_HEADER}"
  "${OUTPUT_DF_HEADER}"
)

# python script
set(NBL_GEN_PY "${CMAKE_CURRENT_SOURCE_DIR}/gen.py")

# custom command
set(NBL_COMMAND
  "${_Python3_EXECUTABLE}" "${NBL_GEN_PY}" "${INPUT_DL_JSON}" "${INPUT_DF_JSON}" "${OUTPUT_DL_HEADER}" "${OUTPUT_DF_HEADER}"
)

add_custom_command(OUTPUT ${OUTPUT_HEADERS}
    DEPENDS ${INPUT_JSONS}
    COMMAND ${NBL_COMMAND}
)

add_custom_target(DeviceHeaders DEPENDS ${OUTPUT_HEADERS})