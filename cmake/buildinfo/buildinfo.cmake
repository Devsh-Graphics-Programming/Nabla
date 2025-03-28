cmake_host_system_information(RESULT CPU_NAME QUERY PROCESSOR_NAME)
cmake_host_system_information(RESULT CPU_DESC QUERY PROCESSOR_DESCRIPTION)

if (NOT DEFINED NBL_ROOT_PATH_BINARY)
	message(FATAL_ERROR "NBL_ROOT_PATH_BINARY is not defined or it's empty")
endif

add_custom_command(
	OUTPUT cmake_info.json
	COMMAND ${CMAKE_COMMAND} -E capabilities >> cmake_info.json
	COMMENT "Generating ${NBL_ROOT_PATH_BINARY}/cmake_info.json"
)
if (Vulkan_FOUND)
	set(VKSDK_INFO_COMMAND "$ENV{VULKAN_SDK}/bin/vulkaninfoSDK")
	add_custom_command(
		OUTPUT vulkan_info.json
		COMMAND ${VKSDK_INFO_COMMAND} -j -o ${NBL_ROOT_PATH_BINARY}/vulkan_info.json
		COMMENT "Generating ${NBL_ROOT_PATH_BINARY}/vulkan_info.json"
	)
else()
	add_custom_command(
		OUTPUT vulkan_driver_disabled.txt
		COMMAND ${CMAKE_COMMAND} -E touch ${NBL_ROOT_PATH_BINARY}/vulkan_driver_disabled.txt
		COMMENT "Generating ${NBL_ROOT_PATH_BINARY}/vulkan_driver_disabled.txt"
endif()

set(SYSTEM_INFO_DEPENDENCIES 
	cmake_info.json 
	vulkan_info.json
	CMakeCache.txt
	"${NBL_ROOT_PATH_BINARY}/3rdparty/git-version-tracking/nabla_git_info.cpp" 
	"${NBL_ROOT_PATH_BINARY}/3rdparty/git-version-tracking/dxc_git_info.cpp"
)

add_custom_target(system_info 
	DEPENDS ${SYSTEM_INFO_DEPENDENCIES}
	COMMAND ${CMAKE_COMMAND} -E tar c build_info.zip --format=zip ${SYSTEM_INFO_DEPENDENCIES}
	COMMENT "Generating ${NBL_ROOT_PATH_BINARY}/build_info.zip"
)