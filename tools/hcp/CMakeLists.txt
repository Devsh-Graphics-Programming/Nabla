nbl_create_executable_project("" "" "" "")
add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PRIVATE $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)

enable_testing()

set(NBL_HCP_CI_ARGS
	--buffer-bytes 67108864
	--seed 12345
)

function(nbl_hcp_add_ci_test mode)
	string(TOUPPER "${mode}" mode_upper)
	add_test(NAME "NBL_HCP_${mode_upper}"
		COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --runtime-tuning "${mode}" ${NBL_HCP_CI_ARGS}
		WORKING_DIRECTORY "$<TARGET_FILE_DIR:${EXECUTABLE_NAME}>"
		COMMAND_EXPAND_LISTS
	)
endfunction()

nbl_hcp_add_ci_test(sequential)
nbl_hcp_add_ci_test(heuristic)
nbl_hcp_add_ci_test(hybrid)

set_tests_properties(
	NBL_HCP_SEQUENTIAL
	NBL_HCP_HEURISTIC
	NBL_HCP_HYBRID
	PROPERTIES
		LABELS "hash;ci"
)
