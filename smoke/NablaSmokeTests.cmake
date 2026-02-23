function(nabla_smoke_add_install_load_api_test)
  set(_nbl_smoke_options LEGACY_CTEST_MODE)
  set(_nbl_smoke_one_value_args TEST_NAME EXE_PATH CRASH_HANDLER_SCRIPT ENABLE_CRASH_HANDLER)
  set(_nbl_smoke_multi_value_args ENVIRONMENT)
  cmake_parse_arguments(_NBL_SMOKE "${_nbl_smoke_options}" "${_nbl_smoke_one_value_args}" "${_nbl_smoke_multi_value_args}" ${ARGN})

  if(NOT _NBL_SMOKE_TEST_NAME)
    message(FATAL_ERROR "nabla_smoke_add_install_load_api_test requires TEST_NAME")
  endif()
  if(NOT _NBL_SMOKE_EXE_PATH)
    message(FATAL_ERROR "nabla_smoke_add_install_load_api_test requires EXE_PATH")
  endif()

  if(WIN32 AND _NBL_SMOKE_ENABLE_CRASH_HANDLER)
    if(_NBL_SMOKE_LEGACY_CTEST_MODE)
      add_test("${_NBL_SMOKE_TEST_NAME}"
        powershell -NoProfile -ExecutionPolicy Bypass
        -File "${_NBL_SMOKE_CRASH_HANDLER_SCRIPT}"
        -Exe "${_NBL_SMOKE_EXE_PATH}"
      )
    else()
      add_test(NAME "${_NBL_SMOKE_TEST_NAME}" COMMAND
        powershell -NoProfile -ExecutionPolicy Bypass
        -File "${_NBL_SMOKE_CRASH_HANDLER_SCRIPT}"
        -Exe "${_NBL_SMOKE_EXE_PATH}"
      )
    endif()
  else()
    if(_NBL_SMOKE_LEGACY_CTEST_MODE)
      add_test("${_NBL_SMOKE_TEST_NAME}" "${_NBL_SMOKE_EXE_PATH}")
    else()
      add_test(NAME "${_NBL_SMOKE_TEST_NAME}" COMMAND "${_NBL_SMOKE_EXE_PATH}")
    endif()
  endif()

  if(_NBL_SMOKE_ENVIRONMENT)
    set_tests_properties("${_NBL_SMOKE_TEST_NAME}" PROPERTIES ENVIRONMENT "${_NBL_SMOKE_ENVIRONMENT}")
  endif()
endfunction()
