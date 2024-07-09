set(NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT "${THIRD_PARTY_SOURCE_DIR}/imgui_test_engine")
set(NBL_IMGUI_TEST_ENGINE_ROOT "${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}/imgui_test_engine")
set(NBL_IMGUI_TEST_SUITE_ROOT "${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}/imgui_test_suite")

set(NBL_EXTRA_INCLUDES
	"${NBL_IMGUI_TEST_ENGINE_PROJECT_ROOT}"
	"${NBL_IMGUI_TEST_ENGINE_ROOT}"
	"${NBL_IMGUI_TEST_SUITE_ROOT}"
	$<TARGET_PROPERTY:imgui,INCLUDE_DIRECTORIES>
	$<TARGET_PROPERTY:implot,INCLUDE_DIRECTORIES>
)

file(GLOB_RECURSE NBL_EXTRA_SOURCES CONFIGURE_DEPENDS "${NBL_IMGUI_TEST_ENGINE_ROOT}/*.cpp")
file(GLOB_RECURSE NBL_TEST_SUITE_SOURCES CONFIGURE_DEPENDS "${NBL_IMGUI_TEST_SUITE_ROOT}/imgui_test*.cpp")
file(GLOB_RECURSE NBL_TEST_SUITE_SOURCES CONFIGURE_DEPENDS "${NBL_IMGUI_TEST_SUITE_ROOT}/imgui_test*.cpp")

add_library(imtestsuite STATIC
	${NBL_TEST_SUITE_SOURCES}
)
target_include_directories(imtestsuite 
	PUBLIC ${NBL_EXTRA_INCLUDES}
)
nbl_handle_runtime_lib_properties(imtestsuite)
set_property(TARGET imtestsuite PROPERTY CXX_STANDARD 14) # NOTE! THOSE TESTS DO NOT COMPILE WITH HIGHER STANDARDS SO WE WRAP SOURCES INTO LIBRARY COMPILED WITH LOWER ONE

target_precompile_headers(imtestsuite
	PUBLIC "${NBL_IMGUI_TEST_SUITE_ROOT}/imgui_test_suite_imconfig.h" # force include config header into test suite sources
)
target_link_libraries(imtestsuite PUBLIC implot)

set(NBL_EXTRA_OPTIONS
	# TODO: add if required
) 

set(NBL_EXTRA_LIBS
	imgui
	imtestsuite
	"${NBL_EXT_IMGUI_UI_LIB}" # Nabla IMGUI backend
)

nbl_create_executable_project("${NBL_EXTRA_SOURCES}" "${NBL_EXTRA_OPTIONS}" "${NBL_EXTRA_INCLUDES}" "${NBL_EXTRA_LIBS}")