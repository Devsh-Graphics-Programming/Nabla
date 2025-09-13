include_guard(GLOBAL)

if (NOT DEFINED NBL_ROOT_PATH_BINARY)
	message(FATAL_ERROR "NBL_ROOT_PATH_BINARY is not defined or it's empty")
endif()

if(NOT TARGET gtml)
	message(FATAL_ERROR "gtml target not defined!")
endif()

set(OUTPUT_DIR "${NBL_ROOT_PATH_BINARY}")

if(Vulkan_FOUND)
	set(VKSDK_INFO_CMD "${VULKAN_SDK}/bin/vulkaninfoSDK" -j -o "${OUTPUT_DIR}/vulkan-info.json")
else()
	set(VKSDK_INFO_CMD "${CMAKE_COMMAND}" -E touch "${OUTPUT_DIR}/vulkan-info.json")
endif()

execute_process(COMMAND ${VKSDK_INFO_CMD})
execute_process(COMMAND "${CMAKE_COMMAND}" -E capabilities OUTPUT_VARIABLE PIPE)
file(WRITE "${OUTPUT_DIR}/cmake-caps.json" "${PIPE}")

get_target_property(GTML_SOURCES gtml SOURCES)
list(FILTER GTML_SOURCES INCLUDE REGEX "git_info\\.cpp$")

set(BUILD_INFO_DEPENDENCIES 
	"${OUTPUT_DIR}/vulkan-info.json"
	"${OUTPUT_DIR}/cmake-caps.json"
	CMakeCache.txt
	CMakeFiles/CMakeConfigureLog.yaml
	${GTML_SOURCES}
)

add_custom_target(nbl_build_info 
	DEPENDS ${BUILD_INFO_DEPENDENCIES} gtml
	COMMAND "${CMAKE_COMMAND}" -E tar c build_info.zip --format=zip ${BUILD_INFO_DEPENDENCIES}
	COMMENT "Generating \"${NBL_ROOT_PATH_BINARY}/build_info.zip\", attach this within your issue"
)