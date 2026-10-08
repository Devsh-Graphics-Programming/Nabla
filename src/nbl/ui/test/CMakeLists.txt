# Regression test for the XCB window manager threading rules. Needs an X server (DISPLAY) at run time, no GPU.
# Run with `ctest --test-dir <build>/src/nbl/ui/test`; it's reported as skipped without an X server.
enable_testing()

add_executable(nbl_ui_xcb_window_manager_test CWindowManagerXcbTest.cpp)
target_link_libraries(nbl_ui_xcb_window_manager_test PRIVATE Nabla)
target_compile_features(nbl_ui_xcb_window_manager_test PRIVATE cxx_std_20)

add_test(NAME NBL_UI_XCB_WINDOW_MANAGER COMMAND $<TARGET_FILE:nbl_ui_xcb_window_manager_test>)
set_tests_properties(NBL_UI_XCB_WINDOW_MANAGER PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
