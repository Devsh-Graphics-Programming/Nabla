include(common RESULT_VARIABLE RES)
if(NOT RES)
    message(FATAL_ERROR "common.cmake not found. Should be in {repo_root}/cmake directory")
endif()

set(ASSET_DIR ${PROJECT_SOURCE_DIR}/examples_tests/media)
nbl_create_executable_project(../../src/nbl/ext/DebugDraw/CDraw3DLine.cpp "" "" "" ${ASSET_DIR})
