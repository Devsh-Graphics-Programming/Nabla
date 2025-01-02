nbl_create_executable_project("" "" "" "")

enable_testing()

set(GODBOLT_BINARY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compiler-explorer")
set(GODBOLT_BINARY_PRETEST_DIRECTORY "${GODBOLT_BINARY_DIRECTORY}/pre-test")
set(NBL_NSC_COMPILE_DIRECTORY "${GODBOLT_BINARY_PRETEST_DIRECTORY}/.compile/$<CONFIG>")
set(NBL_DOCKER_CT_NSC_VOLUME_SOURCE "${GODBOLT_BINARY_DIRECTORY}/install")

add_test(NAME NBL_NSC_INSTALL_RUNTIMES_TEST
	COMMAND "${CMAKE_COMMAND}" --install . --config $<CONFIG> --prefix "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}" --component Runtimes
	WORKING_DIRECTORY "${NBL_ROOT_PATH_BINARY}"
	COMMAND_EXPAND_LISTS
)

add_test(NAME NBL_NSC_INSTALL_EXECUTABLES_TEST
	COMMAND "${CMAKE_COMMAND}" --install . --config $<CONFIG> --prefix "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}" --component Executables
	WORKING_DIRECTORY "${NBL_ROOT_PATH_BINARY}"
	COMMAND_EXPAND_LISTS
)

get_target_property(NBL_PACKAGE_RUNTIME_EXE_DIR_PATH ${EXECUTABLE_NAME} NBL_PACKAGE_RUNTIME_EXE_DIR_PATH)

set(NBL_NSC_COMPILE_COMMAND
	-Fc "${NBL_NSC_COMPILE_DIRECTORY}/output.spv"
	-spirv -Zpr -enable-16bit-types -fvk-use-scalar-layout -Wno-c++11-extensions -Wno-c++1z-extensions -Wno-c++14-extensions -Wno-gnu-static-float-init -fspv-target-env=vulkan1.3 -HV 202x -E main -fspv-debug=source -fspv-debug=tool -T cs_6_7
	"${CMAKE_CURRENT_SOURCE_DIR}/test/hlsl/input.hlsl"
)

add_test(NAME NBL_NSC_COMPILE_AT_EXE_CWD_TEST
	COMMAND "$<TARGET_FILE_BASE_NAME:${EXECUTABLE_NAME}>" ${NBL_NSC_COMPILE_COMMAND}
	WORKING_DIRECTORY "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}/${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}"
	COMMAND_EXPAND_LISTS
)

add_test(NAME NBL_NSC_COMPILE_CUSTOM_CWD_TEST
	COMMAND "${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}/$<TARGET_FILE_BASE_NAME:${EXECUTABLE_NAME}>" ${NBL_NSC_COMPILE_COMMAND}
	WORKING_DIRECTORY "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}"
	COMMAND_EXPAND_LISTS
)

set(NBL_DOCKER_NSC_COMPILER_CONFIG_OUTPUT "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}/hlsl.local.properties.cmake")

set(NBL_DOCKER_CT_NSC_VOLUME_TARGET "C:\\nsc\\install")
string(GENEX_STRIP "${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}" NBL_RELATIVE_ENTRY)
set(NSC_RELEASE_EXECUTABLE "${NBL_DOCKER_CT_NSC_VOLUME_TARGET}/${NBL_RELATIVE_ENTRY}/${EXECUTABLE_NAME}")
set(NSC_DEBUG_EXECUTABLE "${NBL_DOCKER_CT_NSC_VOLUME_TARGET}/debug/${NBL_RELATIVE_ENTRY}/${EXECUTABLE_NAME}_d")
cmake_path(NATIVE_PATH NSC_RELEASE_EXECUTABLE NORMALIZE NSC_RELEASE_EXECUTABLE)
cmake_path(NATIVE_PATH NSC_DEBUG_EXECUTABLE NORMALIZE NSC_DEBUG_EXECUTABLE)

# TODO: NABLA_REV_TARGET

string(APPEND CONFIG_CONTENT
[=[
compilers=&dxc

defaultCompiler=nsc_release_upstream
supportsBinary=true
supportsBinaryObject=true
compilerType=nsc-spirv
needsMulti=false
supportsLibraryCodeFilter=true
disassemblerPath=@SPIRV_DIS_EXE@
demangler=

group.dxc.compilers=nsc_release_upstream:nsc_debug_upstream
group.dxc.includeFlag=-I
group.dxc.versionFlag=--version
group.dxc.groupName=NSC compilers

compiler.nsc_release_upstream.exe=@NSC_RELEASE_EXECUTABLE@.exe
compiler.nsc_release_upstream.name=NSC (Release)
compiler.nsc_release_upstream.notification=The NSC (Release) has been compiled from following <a href="https://github.com/Devsh-Graphics-Programming/Nabla/commit/@NABLA_REV_TARGET@" target="_blank" rel="noopener noreferrer">commit<sup><small class="fas fa-external-link-alt opens-new-window" title="Opens the commit in a new window"></small></sup></a>.
compiler.nsc_release_upstream.supportsExecute=false
compiler.nsc_release_upstream.options=
compiler.nsc_release_upstream.disassemblerPath=@SPIRV_DIS_EXE@
compiler.nsc_release_upstream.demangler=

compiler.nsc_debug_upstream.exe=@NSC_DEBUG_EXECUTABLE@.exe
compiler.nsc_debug_upstream.name=NSC (Debug)
compiler.nsc_debug_upstream.notification=The NSC (Debug) has been compiled from following <a href="https://github.com/Devsh-Graphics-Programming/Nabla/commit/@NABLA_REV_TARGET@" target="_blank" rel="noopener noreferrer">commit<sup><small class="fas fa-external-link-alt opens-new-window" title="Opens the commit in a new window"></small></sup></a>.
compiler.nsc_debug_upstream.supportsExecute=false
compiler.nsc_debug_upstream.options=
compiler.nsc_debug_upstream.disassemblerPath=@SPIRV_DIS_EXE@
compiler.nsc_debug_upstream.demangler=
]=]
)

find_program(SPIRV_DIS_EXE
	NAMES spirv-dis
	HINTS "$ENV{VULKAN_SDK_INSTALL_DIRECTORY}/Bin"
  HINTS "$ENV{VK_SDK_PATH}/Bin"
  HINTS "$ENV{VULKAN_SDK}/Bin"
	REQUIRED
)

cmake_path(GET SPIRV_DIS_EXE PARENT_PATH VULKAN_SDK_BIN_DIRECTORY)
cmake_path(GET SPIRV_DIS_EXE FILENAME SPIRV_DIS_EXE)
set(SPIRV_DIS_EXE "C:\\vulkan\\bin\\${SPIRV_DIS_EXE}")
cmake_path(NATIVE_PATH SPIRV_DIS_EXE NORMALIZE SPIRV_DIS_EXE)

string(CONFIGURE "${CONFIG_CONTENT}" CONFIG_CONTENT @ONLY)
file(WRITE "${NBL_DOCKER_NSC_COMPILER_CONFIG_OUTPUT}" "${CONFIG_CONTENT}")

set(NBL_DOCKER_CE_COMPOSE_BASE "${NBL_ROOT_PATH}/docker/compiler-explorer/compose.yml")
set(NBL_DOCKER_CE_COMPOSE_TARGET "${GODBOLT_BINARY_DIRECTORY}/compose.yml")

include(InstallRequiredSystemLibraries)

string(APPEND COMPOSE_CONTENT
[=[
services:
  compiler-explorer-nsc:
    extends:
        file: @NBL_DOCKER_CE_COMPOSE_BASE@
        service: compiler-explorer
    container_name: dev.ce.nsc
    entrypoint:
      - "cmd"
      - "/c"
      - >
        C:\\redist\\vc_redist.x64.exe /quiet /install
        && copy C:\\nsc\\install\\hlsl.local.properties.cmake %GIT_GODBOLT_REPOSITORY_PATH%\\etc\\config\\hlsl.local.properties
        && npm --prefix %GIT_GODBOLT_REPOSITORY_PATH% run dev -- --language hlsl
    volumes:
      - type: bind
        source: .\install
        target: @NBL_DOCKER_CT_NSC_VOLUME_TARGET@
      - type: bind
        source: @MSVC_REDIST_DIR@
        target: C:\redist
        read_only: true
      - type: bind
        source: @VULKAN_SDK_BIN_DIRECTORY@
        target: C:\vulkan\bin
        read_only: true

networks:
  docker_default:
    external: true
]=]
)

string(CONFIGURE "${COMPOSE_CONTENT}" COMPOSE_CONTENT @ONLY)
file(WRITE "${NBL_DOCKER_CE_COMPOSE_TARGET}" "${COMPOSE_CONTENT}")

add_custom_target(run-compiler-explorer
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --blue "Performing Pre-Test..."
    COMMAND ctest -C $<CONFIG> --stop-on-failure
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --yellow "Attempt to remove any remaining previous container..."
    COMMAND docker compose -f "${NBL_DOCKER_CE_COMPOSE_TARGET}" rm -f
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Starting Compiler Explorer!"
    COMMAND docker compose -f "${NBL_DOCKER_CE_COMPOSE_TARGET}" up --build compiler-explorer-nsc
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
add_dependencies(run-compiler-explorer nsc)
set_target_properties(run-compiler-explorer PROPERTIES FOLDER "Utilities")