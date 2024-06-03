set(SRC_VIDEO_PATH ${CMAKE_SOURCE_DIR}/src/nbl/video)
set(INPUT_JSON ${SRC_VIDEO_PATH}/device_limits.json ${SRC_VIDEO_PATH}/device_features.json)
set(OUTPUT_HEADER ${SRC_VIDEO_PATH}/test_device_limits.h ${SRC_VIDEO_PATH}/test_device_features.h)

add_custom_command(
    OUTPUT  ${OUTPUT_HEADER}
    DEPENDS ${INPUT_JSON}
    COMMAND "${_Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/gen.py" ${INPUT_JSON} ${OUTPUT_HEADER}
)

add_custom_target(DeviceHeaders
  DEPENDS ${OUTPUT_HEADER}
)