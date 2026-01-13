nbl_create_executable_project("" "" "" "")

enable_testing()

add_dependencies(${EXECUTABLE_NAME} argparse)
target_include_directories(${EXECUTABLE_NAME} PRIVATE $<TARGET_PROPERTY:argparse,INTERFACE_INCLUDE_DIRECTORIES>)

set(GODBOLT_BINARY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compiler-explorer")
set(GODBOLT_BINARY_PRETEST_DIRECTORY "${GODBOLT_BINARY_DIRECTORY}/pre-test")
set(NBL_NSC_COMPILE_DIRECTORY "${GODBOLT_BINARY_PRETEST_DIRECTORY}/.compile/$<CONFIG>")
set(NBL_NSC_PREINSTALL_DIRECTORY "${GODBOLT_BINARY_PRETEST_DIRECTORY}/.preinstall")
make_directory("${NBL_NSC_PREINSTALL_DIRECTORY}")

set(NBL_DOCKER_CT_NSC_VOLUME_SOURCE "${GODBOLT_BINARY_DIRECTORY}/install")
set(NBL_DOCKER_CTX_DIR "${GODBOLT_BINARY_DIRECTORY}/.ctx")
make_directory("${NBL_DOCKER_CTX_DIR}")

set(NBL_DOCKER_INSTALL_BAT_FILENAME install-production.bat)
set(NBL_DOCKER_CT_NSC_INSTALL_BAT "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}/${NBL_DOCKER_INSTALL_BAT_FILENAME}")

add_test(NAME NBL_NSC_INSTALL_RUNTIMES_TEST
	COMMAND "${CMAKE_COMMAND}" --install . --config $<CONFIG> --prefix "${NBL_NSC_PREINSTALL_DIRECTORY}" --component Runtimes
	WORKING_DIRECTORY "${NBL_ROOT_PATH_BINARY}"
	COMMAND_EXPAND_LISTS
)

add_test(NAME NBL_NSC_INSTALL_EXECUTABLES_TEST
	COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_CURRENT_BINARY_DIR}" --config $<CONFIG> --prefix "${NBL_NSC_PREINSTALL_DIRECTORY}" --component Executables
	COMMAND_EXPAND_LISTS
)

get_target_property(NBL_PACKAGE_RUNTIME_EXE_DIR_PATH ${EXECUTABLE_NAME} NBL_PACKAGE_RUNTIME_EXE_DIR_PATH)

set(NBL_NSC_COMPILE_COMMAND
	-Fc "${NBL_NSC_COMPILE_DIRECTORY}/output.spv"
	-spirv -Zpr -enable-16bit-types -fvk-use-scalar-layout -Wno-c++11-extensions -Wno-c++1z-extensions -Wno-c++14-extensions -Wno-gnu-static-float-init -fspv-target-env=vulkan1.3 -HV 202x -E main -fspv-debug=source -fspv-debug=tool -T cs_6_7
	"${CMAKE_CURRENT_SOURCE_DIR}/test/hlsl/input.hlsl"
)

set(NBL_NSC_PREINSTALL_TARGET_EXE_DIRECTORY "${NBL_NSC_PREINSTALL_DIRECTORY}/${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}")
set(NBL_NSC_PREINSTALL_TARGET_EXE_FILENAME $<TARGET_FILE_NAME:${EXECUTABLE_NAME}>)
set(NBL_NSC_PREINSTALL_TARGET_EXE_FILEPATH "${NBL_NSC_PREINSTALL_TARGET_EXE_DIRECTORY}/${NBL_NSC_PREINSTALL_TARGET_EXE_FILENAME}")
set(NBL_NSC_BUILD_INFO_FILENAME build-info.json)
set(NBL_NSC_PREINSTALL_TARGET_BUILD_INFO "${NBL_NSC_PREINSTALL_TARGET_EXE_DIRECTORY}/${NBL_NSC_BUILD_INFO_FILENAME}")

add_test(NAME NBL_NSC_COMPILE_AT_EXE_CWD_TEST
	COMMAND "${NBL_NSC_PREINSTALL_TARGET_EXE_FILENAME}" ${NBL_NSC_COMPILE_COMMAND}
	WORKING_DIRECTORY "${NBL_NSC_PREINSTALL_TARGET_EXE_DIRECTORY}"
	COMMAND_EXPAND_LISTS
)

add_test(NAME NBL_NSC_COMPILE_CUSTOM_CWD_TEST
	COMMAND "${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}/${NBL_NSC_PREINSTALL_TARGET_EXE_FILENAME}" ${NBL_NSC_COMPILE_COMMAND}
	WORKING_DIRECTORY "${NBL_NSC_PREINSTALL_DIRECTORY}"
	COMMAND_EXPAND_LISTS
)

add_test(NAME NBL_NSC_DUMP_BUILD_INFO_TEST
  COMMAND "${NBL_NSC_PREINSTALL_TARGET_EXE_FILEPATH}" --dump-build-info --file "${NBL_NSC_PREINSTALL_TARGET_BUILD_INFO}"
  COMMAND_EXPAND_LISTS
)

if(NBL_ENABLE_DOCKER_INTEGRATION)

find_program(DOCKER_EXE NAMES docker REQUIRED)
set(BASE_IMAGE ghcr.io/devsh-graphics-programming/compiler-explorer-docker:nano-2022)
set(CORE_IMAGE mcr.microsoft.com/windows/servercore:ltsc2022)

function(PROMOTE_PROCESS_ISOLATION BASE VAR)
    set(${VAR} True)
    
    macro(INSPECT IMAGE)
      execute_process(COMMAND "${DOCKER_EXE}" inspect --format={{.OsVersion}} ${IMAGE} 
        RESULT_VARIABLE INSPECTION_OK 
        OUTPUT_VARIABLE TARGET_KERNEL 
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
    endmacro()

    macro(TO_PROCESS IMAGE TARGET_KERNEL)
      execute_process(COMMAND "${DOCKER_EXE}" run --rm --isolation process --entrypoint cmd ${BASE} /K 
        RESULT_VARIABLE PROCESS_ISOLATION_OK 
        OUTPUT_QUIET ERROR_QUIET
      )

      if(${PROCESS_ISOLATION_OK} EQUAL 0)
          message(STATUS "Promoting \"${IMAGE}\" [${TARGET_KERNEL}] to process isolation")
      else()
          set(${VAR} False)
          message(STATUS "Cannot promote \"${IMAGE}\" [${TARGET_KERNEL}] to process isolation, requires falling back to HyperV. Please update your docker host OS.")
      endif()
    endmacro()

    INSPECT(${BASE})

    if(${INSPECTION_OK} EQUAL 0)
      TO_PROCESS(${BASE} ${TARGET_KERNEL})
    else()
      message(STATUS "\"${BASE}\" not found in local registry, pulling...")
      execute_process(COMMAND "${DOCKER_EXE}" pull ${BASE})

      INSPECT(${BASE})
      TO_PROCESS(${BASE} ${TARGET_KERNEL})
    endif()

    set(${VAR} ${${VAR}} PARENT_SCOPE)
endfunction()

PROMOTE_PROCESS_ISOLATION(${BASE_IMAGE} USE_PROCESS_ISOLATION)

if(NOT USE_PROCESS_ISOLATION)
  # NOTE: we would need to use GET_RUNTIME_DEPENDENCIES which uses objdump
  # https://cmake.org/cmake/help/latest/command/file.html#get-runtime-dependencies
  # to collect *all* missing deps and copy (FROM at least server core) to destination nano
  # image, it will fail currently if we fully isolate it with VM due to lack of certain DLLs
  # BUT it means violating EULA, hence we are not going to support it, also (**)
  message(FATAL_ERROR "HyperV is NOT supported! Update your OS!")
endif()

function(GET_LABEL BASE_IMAGE LABEL VAR)
    set(FORMAT "{{ index .Config.Labels \"${LABEL}\" }}")
    execute_process(COMMAND ${DOCKER_EXE} inspect --format=${FORMAT} ${BASE_IMAGE}
        OUTPUT_VARIABLE OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_VARIABLE ERR
        RESULT_VARIABLE RES
    )

    if(NOT RES EQUAL 0)
      message(WARNING "Could not get \"${LABEL}\" label from \"${BASE_IMAGE}\" image, it doesn't exist!")
    endif()

    set(${VAR} "${OUT}" PARENT_SCOPE)
endfunction()

GET_LABEL(${BASE_IMAGE} org.opencontainers.image.title ORG_LABEL_TITLE)
GET_LABEL(${BASE_IMAGE} org.opencontainers.image.source ORG_LABEL_SOURCE)
GET_LABEL(${BASE_IMAGE} org.opencontainers.image.description ORG_LABEL_DESCRIPTION)

find_program(CTEST_EXE NAMES ctest REQUIRED)
find_file(DXIL_DLL NAMES dxil.dll HINTS "$ENV{CMAKE_WINDOWS_KITS_10_DIR}/Redist/D3D/x64" "C:/Program Files (x86)/Windows Kits/10/Redist/D3D/x64" REQUIRED)

set(ICU_GLOBALIZATION_DIR C:\\Windows\\Globalization\\ICU)
find_file(UCRTBASED_DLL NAMES ucrtbased.dll HINTS ${UCRTBASED_DLL_DIR} REQUIRED)

find_program(SPIRV_DIS_EXE NAMES spirv-dis HINTS "${VULKAN_SDK}/Bin" REQUIRED)
cmake_path(GET SPIRV_DIS_EXE PARENT_PATH SPIRV_DIS_DIR)
cmake_path(NATIVE_PATH SPIRV_DIS_DIR NORMALIZE SPIRV_DIS_DIR)

if(MSVC_REDIST_BASE) # fallback to our toolset
  set(MSVC_REDIST_DIR "${MSVC_REDIST_BASE}")
else()
  include(InstallRequiredSystemLibraries)
  if(NOT MSVC_REDIST_DIR)
    message(FATAL_ERROR "Could not find MSVC_REDIST_DIR, define yourself!")
  endif()
endif()

cmake_path(NATIVE_PATH MSVC_REDIST_DIR NORMALIZE TOOLSET_REDIST_PATH)

file(GLOB_RECURSE VC_MODULES LIST_DIRECTORIES false
  "${TOOLSET_REDIST_PATH}/x64/*.CRT/*.dll"
  "${TOOLSET_REDIST_PATH}/debug_nonredist/x64/*.DebugCRT/*.dll"
)

if(NOT VC_MODULES)
  message(FATAL_ERROR "Failed to GLOB for VC Redist modules!")
endif()

make_directory("${NBL_DOCKER_CTX_DIR}/Runtimes")
make_directory("${NBL_DOCKER_CTX_DIR}/Nabla")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${DXIL_DLL}" "${NBL_DOCKER_CTX_DIR}/Runtimes"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${UCRTBASED_DLL}" "${NBL_DOCKER_CTX_DIR}/Runtimes"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${SPIRV_DIS_EXE}" "${NBL_DOCKER_CTX_DIR}/Runtimes"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${VC_MODULES} "${NBL_DOCKER_CTX_DIR}/Runtimes"
)

set(CT_RUNTIMES C:/runtimes)
cmake_path(NATIVE_PATH CT_RUNTIMES NORMALIZE CT_RUNTIMES)

set(NBL_DOCKER_CT_NSC_VOLUME_TARGET "${CT_RUNTIMES}/Nabla")
cmake_path(NATIVE_PATH NBL_DOCKER_CT_NSC_VOLUME_SOURCE NORMALIZE NBL_DOCKER_CT_NSC_VOLUME_SOURCE)
cmake_path(NATIVE_PATH NBL_DOCKER_CT_NSC_VOLUME_TARGET NORMALIZE NBL_DOCKER_CT_NSC_VOLUME_TARGET)
cmake_path(NATIVE_PATH NBL_NSC_PREINSTALL_DIRECTORY NORMALIZE NBL_NSC_PREINSTALL_DIRECTORY)

string(CONFIGURE [=[
# syntax=docker/dockerfile:1
# escape=`

# ---------------- COMPRESS STEP ----------------
FROM @BASE_IMAGE@ as compress

COPY --link Runtimes/ C:/pack/Windows/System32/
COPY --link Nabla/ C:/pack/runtimes/Nabla/

ARG IMPL_COMPRESSION_OPTIONS=-T0
ARG IMPL_COMPRESSION_LEVEL=3

WORKDIR C:\pack
RUN `
tar -cf - Windows | zstd %IMPL_COMPRESSION_OPTIONS% -%IMPL_COMPRESSION_LEVEL% -o windows-artifacts.tar.zst && `
tar -cf - runtimes | zstd %IMPL_COMPRESSION_OPTIONS% -%IMPL_COMPRESSION_LEVEL% -o nabla-artifacts.tar.zst

# ---------------- FINAL IMAGE ----------------
FROM @BASE_IMAGE@

COPY --link --from=compress ["C:/pack/windows-artifacts.tar.zst", "C:/pack/"]
COPY --link --from=compress ["C:/pack/nabla-artifacts.tar.zst", "C:/pack/"]
COPY hlsl.local.properties.cmake C:/Compiler-Explorer/etc/config/hlsl.local.properties

ENV NBL_INSTALL_DIRECTORY=@NBL_DOCKER_CT_NSC_VOLUME_TARGET@ `
NBL_EXPLICIT_MODULE_LOAD_LOG=ON

WORKDIR C:/Compiler-Explorer
ENTRYPOINT [ `
  "C:\\unpack.bat", "&&", `
  "copy", "C:\\mount\\Windows\\System32\\icu.dll", "C:\\Windows\\System32\\icu.dll", "&&", `
  "node", "--no-warnings", "--no-deprecation", "--import=tsx", "./app.js", "--language", "hlsl" `
]

LABEL org.opencontainers.image.title="[Nabla Shader Compiler (NSC)]: @ORG_LABEL_TITLE@"
LABEL org.opencontainers.image.source=https://github.com/Devsh-Graphics-Programming/Nabla
LABEL org.opencontainers.image.description="[Nabla Shader Compiler (NSC)]: @ORG_LABEL_DESCRIPTION@"

]=] INSTRUCTIONS @ONLY)

set(DOCKERFILE "${NBL_DOCKER_CTX_DIR}/Dockerfile")
file(WRITE "${DOCKERFILE}" "${INSTRUCTIONS}")

if(DEFINED ENV{NSC_IMAGE_NAME})
  set(NSC_IMAGE_NAME "$ENV{NSC_IMAGE_NAME}")
else()
  set(NSC_IMAGE_NAME nano/godbolt/nsc)
endif()

set(NBL_DOCKER_NSC_COMPILER_CONFIG_OUTPUT "${NBL_DOCKER_CTX_DIR}/hlsl.local.properties.cmake")
string(GENEX_STRIP "${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}" NBL_RELATIVE_ENTRY)
set(OUTPUT_CONFIG_FILE $<PATH:NORMAL_PATH,${NBL_DOCKER_NSC_COMPILER_CONFIG_OUTPUT}>)

set(ICU_DIR C:\\Windows\\Globalization\\ICU)
set(ICU_DLL C:\\Windows\\System32\\icu.dll)
if(NOT EXISTS ${ICU_DIR} OR NOT EXISTS ${ICU_DLL})
  # fallback for CI purposes, NOTE: we do NOT distribute those in final image as we have host runner requirements (**)
  message(STATUS "\"${ICU_DIR}\" or \"${ICU_DLL}\ not found, fallback: copying them to the runner from \"${CORE_IMAGE}\"")
  execute_process(COMMAND "${DOCKER_EXE}" rm -f nano-orphan RESULT_VARIABLE res)
  execute_process(COMMAND "${DOCKER_EXE}" run -di --isolation process --name nano-orphan --entrypoint cmd ${CORE_IMAGE} COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${DOCKER_EXE}" cp nano-orphan:${ICU_DIR} ${ICU_DIR} COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${DOCKER_EXE}" cp nano-orphan:${ICU_DLL} ${ICU_DLL} COMMAND_ERROR_IS_FATAL ANY)
  message(STATUS "Fallback completed, runner patched!")
endif()

set(ORPHAN nsc-orphan)

if(NOT DEFINED NBL_CE_PUBLISH_PORT)
  set(NBL_CE_PUBLISH_PORT 80)
endif()

if(NBL_DOCKER_DIND_BUILD)
  set(NBL_CE_URL http://${ORPHAN}:${NBL_CE_PUBLISH_PORT})
else()
  set(NBL_CE_URL http://localhost:${NBL_CE_PUBLISH_PORT})
endif()

set(NBL_CE_HEALTHY_CHECK_PY "${NBL_ROOT_PATH}/docker/compiler-explorer/ce_healthy_check.py")
set(NBL_CE_ENDPOINT_PY "${NBL_ROOT_PATH}/docker/compiler-explorer/endpoint.py")
set(NBL_NSC_BASIC_HLSL_JPAYLOAD "${CMAKE_CURRENT_SOURCE_DIR}/docker/godbolt/hlsl-basic-compile-payload.json")

# to avoid "too long input" errors we proxy build instructions to CMake script and write it to build directory
string(CONFIGURE [=[
message(STATUS "Killing remaining NSC orphans")
execute_process(COMMAND "@DOCKER_EXE@" 
  rm -f "@ORPHAN@" 
  RESULT_VARIABLE res
)

message(STATUS "Executing CTests")
execute_process(COMMAND "@CTEST_EXE@" 
  -C "$<CONFIG>" --stop-on-failure 
  WORKING_DIRECTORY "@CMAKE_CURRENT_BINARY_DIR@"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Generating NSC build info")
execute_process(COMMAND "@CMAKE_COMMAND@"
  "-DNBL_EXECUTABLE_PATH=@NBL_NSC_PREINSTALL_TARGET_EXE_FILEPATH@"
  "-DNBL_BUILD_INFO=@NBL_NSC_PREINSTALL_TARGET_BUILD_INFO@"
  "-DNBL_OUTPUT_FILE=@NBL_NSC_PREINSTALL_TARGET_BUILD_INFO@"
  "-DNBL_OUTPUT_EXE_OVERRIDE=$<PATH:NORMAL_PATH,@NBL_DOCKER_CT_NSC_VOLUME_TARGET@/@NBL_PACKAGE_RUNTIME_EXE_DIR_PATH@/@NBL_NSC_PREINSTALL_TARGET_EXE_FILENAME@>"
  -P "@NBL_ROOT_PATH@/cmake/scripts/nbl/nablaBuildInfo.cmake"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Generating NSC godbolt config")
execute_process(COMMAND "@CMAKE_COMMAND@"
  "-DSPIRV_DIS_EXE=spirv-dis.exe"
  "-DNSC_RELEASE_BUILD_INFO=$<PATH:NORMAL_PATH,@NBL_NSC_PREINSTALL_DIRECTORY@/@NBL_RELATIVE_ENTRY@/@NBL_NSC_BUILD_INFO_FILENAME@>"
  "-DNSC_RELWITHDEBINFO_BUILD_INFO=$<PATH:NORMAL_PATH,@NBL_NSC_PREINSTALL_DIRECTORY@/relwithdebinfo/@NBL_RELATIVE_ENTRY@/@NBL_NSC_BUILD_INFO_FILENAME@>"
  "-DNSC_DEBUG_BUILD_INFO=$<PATH:NORMAL_PATH,@NBL_NSC_PREINSTALL_DIRECTORY@/debug/@NBL_RELATIVE_ENTRY@/@NBL_NSC_BUILD_INFO_FILENAME@>"
  "-DOUTPUT_CONFIG_FILE=@OUTPUT_CONFIG_FILE@"
  -P "@CMAKE_CURRENT_SOURCE_DIR@/ce-generate-config.cmake"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Updating NSC package context")
execute_process(COMMAND "@CMAKE_COMMAND@" -E copy_directory_if_different
  "$<PATH:NORMAL_PATH,@NBL_NSC_PREINSTALL_DIRECTORY@>"
  "@NBL_DOCKER_CTX_DIR@/Nabla"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Building NSC Godbolt image")
string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ" UTC)
execute_process(COMMAND "@DOCKER_EXE@" build --isolation process
  --label=org.opencontainers.image.created="${BUILD_TIMESTAMP}"
  -f "@DOCKERFILE@" -t @NSC_IMAGE_NAME@ "@NBL_DOCKER_CTX_DIR@"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Running new NSC orphan container")
execute_process(COMMAND "@DOCKER_EXE@" run -di -p @NBL_CE_PUBLISH_PORT@:10240 --isolation process
  --name "@ORPHAN@" --network docker_default
  -v $<PATH:NORMAL_PATH,@ICU_DIR@:@ICU_DIR@:ro>
  -v $<PATH:NORMAL_PATH,C:/Windows/System32:C:/mount/Windows/System32:ro>
  @NSC_IMAGE_NAME@
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Healthy check")
execute_process(COMMAND "@_Python3_EXECUTABLE@" "@NBL_CE_HEALTHY_CHECK_PY@"
  --url "@NBL_CE_URL@" --interval 5 --ticks 12
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Post Basic NSC shader compile check")
execute_process(COMMAND "@_Python3_EXECUTABLE@" "@NBL_CE_ENDPOINT_PY@"
  --url "@NBL_CE_URL@"
  --endpoint /api/compiler/nsc_$<LOWER_CASE:$<CONFIG>>_upstream/compile
  --method POST --json "@NBL_NSC_BASIC_HLSL_JPAYLOAD@"
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "Printing NSC container logs")
execute_process(COMMAND "@DOCKER_EXE@" 
  logs "@ORPHAN@" 
  COMMAND_ERROR_IS_FATAL ANY
)

message(STATUS "OK! NSC container is healthy.")
message(STATUS "Type \"@NBL_CE_URL@\" in your browser to use NSC with Godbolt!")
]=] INSTRUCTIONS @ONLY)

set(SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/run-compiler-explorer-$<CONFIG>.cmake")
file(GENERATE OUTPUT ${SCRIPT_FILE} CONTENT "${INSTRUCTIONS}")

add_custom_target(run-compiler-explorer ALL
    COMMAND "${CMAKE_COMMAND}" -P ${SCRIPT_FILE}
    VERBATIM
    COMMAND_EXPAND_LISTS
)

add_dependencies(run-compiler-explorer nsc)
set_target_properties(run-compiler-explorer PROPERTIES FOLDER "Godbolt")

endif()
