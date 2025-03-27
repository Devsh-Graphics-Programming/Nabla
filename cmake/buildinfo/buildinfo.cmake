# git
find_program(GIT_COMMAND git REQUIRED)

cmake_host_system_information(RESULT CPU_NAME QUERY PROCESSOR_NAME)
cmake_host_system_information(RESULT CPU_DESC QUERY PROCESSOR_DESCRIPTION)

add_custom_command(
	OUTPUT system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "OS: ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "Processor: ${CPU_NAME}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "C Compiler: ${CMAKE_C_COMPILER_ID} ver. ${CMAKE_C_COMPILER_VERSION}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "C Flags: ${CMAKE_C_FLAGS}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "CXX Compiler: ${CMAKE_CXX_COMPILER_ID} ver. ${CMAKE_CXX_COMPILER_VERSION}" >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "NASM: " >> system_info.txt
	COMMAND ${CMAKE_ASM_NASM_COMPILER} -v >> system_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo "VulkanSDK: $ENV{VULKAN_SDK}" >> system_info.txt
	
	COMMENT "Generating ${CMAKE_BINARY_DIR}/system_info.txt"
)

add_custom_command(
	OUTPUT cmake_info.json
	COMMAND ${CMAKE_COMMAND} -E capabilities >> cmake_info.json
	COMMENT "Generating ${CMAKE_BINARY_DIR}/cmake_info.json"
)

set(VKSDK_INFO_COMMAND "$ENV{VULKAN_SDK}/bin/vulkaninfoSDK")
add_custom_command(
	OUTPUT vulkan_info.json
	COMMAND ${VKSDK_INFO_COMMAND} -j -o ${CMAKE_BINARY_DIR}/vulkan_info.json
	COMMENT "Generating ${CMAKE_BINARY_DIR}/vulkan_info.json"
)

add_custom_command(
	OUTPUT git_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "Commit: " >> git_info.txt
	COMMAND ${GIT_COMMAND} diff --quiet || ${CMAKE_COMMAND} -E echo_append "DIRTY " >> git_info.txt
	COMMAND ${GIT_COMMAND} rev-parse HEAD >> git_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "Branch: " >> git_info.txt
	COMMAND ${GIT_COMMAND} rev-parse --abbrev-ref HEAD >> git_info.txt
	COMMAND ${CMAKE_COMMAND} -E echo_append "Tag: " >> git_info.txt
	COMMAND ${GIT_COMMAND} describe --tags >> git_info.txt
	COMMAND ${GIT_COMMAND} diff --quiet || (${CMAKE_COMMAND} -E echo >> git_info.txt && ${GIT_COMMAND} diff >> git_info.txt)
	COMMENT "Generating ${CMAKE_BINARY_DIR}/git_info.txt"
)

set(SYSTEM_INFO_DEPENDENCIES system_info.txt cmake_info.json git_info.txt vulkan_info.json)
add_custom_target(system_info 
	DEPENDS ${SYSTEM_INFO_DEPENDENCIES}
	COMMAND ${CMAKE_COMMAND} -E tar c build_info.zip --format=zip ${SYSTEM_INFO_DEPENDENCIES}
	COMMAND ${CMAKE_COMMAND} -E rm -- ${SYSTEM_INFO_DEPENDENCIES}
	COMMENT "Generating ${CMAKE_BINARY_DIR}/build_info.zip"
)