set(NBL_EXTRA_OPTIONS
	# TODO: add if required
) 

set(NBL_EXTRA_LIBS
	Nabla::ext::ImGUI
)

nbl_create_executable_project("" "${NBL_EXTRA_OPTIONS}" "${NBL_EXTRA_INCLUDES}" "${NBL_EXTRA_LIBS}")

find_package(argparse CONFIG REQUIRED)
find_package(blake3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(gtl CONFIG REQUIRED)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE argparse::argparse BLAKE3::blake3 nlohmann_json::nlohmann_json gtl::gtl)
target_include_directories(${EXECUTABLE_NAME} PUBLIC 
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