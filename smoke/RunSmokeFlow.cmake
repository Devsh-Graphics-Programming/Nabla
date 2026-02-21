if(NOT DEFINED FLOW)
    message(FATAL_ERROR "FLOW is required. Allowed values: CONFIGURE_ONLY, BUILD_ONLY")
endif()

string(TOUPPER "${FLOW}" FLOW)
if(NOT FLOW MATCHES "^(CONFIGURE_ONLY|BUILD_ONLY)$")
    message(FATAL_ERROR "Invalid FLOW='${FLOW}'. Allowed values: CONFIGURE_ONLY, BUILD_ONLY")
endif()

if(NOT DEFINED CONFIG)
    message(FATAL_ERROR "CONFIG is required (e.g. Debug, Release, RelWithDebInfo)")
endif()

if(NOT DEFINED SMOKE_SOURCE_DIR)
    set(SMOKE_SOURCE_DIR "smoke")
endif()

if(NOT DEFINED BUILD_DIR)
    set(BUILD_DIR "smoke/out")
endif()

if(NOT DEFINED INSTALL_DIR)
    set(INSTALL_DIR "${BUILD_DIR}/install")
endif()

if(NOT DEFINED CTEST_BIN)
    if(DEFINED CMAKE_CTEST_COMMAND)
        set(CTEST_BIN "${CMAKE_CTEST_COMMAND}")
    else()
        find_program(CTEST_BIN ctest REQUIRED)
    endif()
endif()

function(run_cmd)
    execute_process(
        COMMAND ${ARGV}
        COMMAND_ECHO STDOUT
        RESULT_VARIABLE _rc
    )
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Command failed with exit code ${_rc}")
    endif()
endfunction()

file(REMOVE_RECURSE "${BUILD_DIR}")

run_cmd(
    "${CMAKE_COMMAND}"
    -S "${SMOKE_SOURCE_DIR}"
    -B "${BUILD_DIR}"
    -D "NBL_SMOKE_FLOW=${FLOW}"
    -D "NBL_SMOKE_INSTALL_SELFTEST=ON"
)

run_cmd(
    "${CMAKE_COMMAND}"
    --build "${BUILD_DIR}"
    --config "${CONFIG}"
)

run_cmd(
    "${CTEST_BIN}"
    --verbose
    --test-dir "${BUILD_DIR}"
    --force-new-ctest-process
    --output-on-failure
    --no-tests=error
    -C "${CONFIG}"
)

file(REMOVE_RECURSE "${INSTALL_DIR}")

run_cmd(
    "${CMAKE_COMMAND}"
    --install "${BUILD_DIR}"
    --config "${CONFIG}"
    --prefix "${INSTALL_DIR}"
)

run_cmd(
    "${CTEST_BIN}"
    --verbose
    --test-dir "${INSTALL_DIR}"
    --force-new-ctest-process
    --output-on-failure
    --no-tests=error
    -C "${CONFIG}"
)
