set(NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT "${THIRD_PARTY_SOURCE_DIR}/imgui_test_engine")
set(NBL_IMGUI_TEST_ENGINE_ROOT "${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}/imgui_test_engine")
set(NBL_IMGUI_TEST_SUITE_ROOT "${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}/imgui_test_suite")

set(NBL_EXTRA_OPTIONS
	# TODO: add if required
) 

set(NBL_EXTRA_LIBS
	imtestengine
	"${NBL_EXT_IMGUI_UI_LIB}" # Nabla IMGUI backend
)

nbl_create_executable_project("" "${NBL_EXTRA_OPTIONS}" "${NBL_EXTRA_INCLUDES}" "${NBL_EXTRA_LIBS}")

add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PUBLIC 
	$<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>
	"${NBL_ROOT_PATH}/examples_tests/common/include"
)

nbl_adjust_flags(MAP_RELEASE Release MAP_RELWITHDEBINFO RelWithDebInfo MAP_DEBUG Debug)
nbl_adjust_definitions()

enable_testing()

add_test(NAME NBL_NITE_RUN_SUITE_BASIC_TESTS
	COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --mode cmd --group test --queued
	COMMAND_EXPAND_LISTS
)

add_test(NAME NBL_NITE_RUN_SUITE_PERF_TESTS
	COMMAND "$<TARGET_FILE:${EXECUTABLE_NAME}>" --mode cmd --group perf --queued
	COMMAND_EXPAND_LISTS
)