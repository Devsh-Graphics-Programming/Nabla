get_filename_component(SRC_VIDEO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/src ABSOLUTE)
get_filename_component(NBL_DEVICE_GEN_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" ABSOLUTE)
set(NBL_DEVICE_GEN_INCLUDE_DIR "${NBL_DEVICE_GEN_INCLUDE_DIR}" CACHE INTERNAL "")
set(NBL_DEVICE_GEN_OUTPUT_DIR "${NBL_DEVICE_GEN_INCLUDE_DIR}/nbl/video")

set(INPUT_JSON ${SRC_VIDEO_PATH}/device_limits.json ${SRC_VIDEO_PATH}/device_features.json)
set(OUTPUT_HEADER ${NBL_DEVICE_GEN_OUTPUT_DIR}/test_device_limits.h ${NBL_DEVICE_GEN_OUTPUT_DIR}/test_device_features.h)

add_custom_command(
    OUTPUT  ${OUTPUT_HEADER}
    DEPENDS ${INPUT_JSON}
    COMMAND "${_Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen.py" ${INPUT_JSON} ${OUTPUT_HEADER}
)

add_custom_target(DeviceHeaders
  DEPENDS ${OUTPUT_HEADER}
)
