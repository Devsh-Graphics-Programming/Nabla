nbl_create_executable_project("" "" "" "")

enable_testing()

set(GODBOLT_BINARY_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/compiler-explorer")
set(GODBOLT_BINARY_PRETEST_DIRECTORY "${GODBOLT_BINARY_DIRECTORY}/pre-test")
set(NBL_NSC_COMPILE_DIRECTORY "${GODBOLT_BINARY_PRETEST_DIRECTORY}/.compile/$<CONFIG>")
set(NBL_NSC_PREINSTALL_DIRECTORY "${GODBOLT_BINARY_PRETEST_DIRECTORY}/.preinstall")

set(NBL_DOCKER_CT_NSC_VOLUME_SOURCE "${GODBOLT_BINARY_DIRECTORY}/install")

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

find_program(DOCKER_EXE
	NAMES docker
	REQUIRED
)

find_program(SPIRV_DIS_EXE
	NAMES spirv-dis
	HINTS "$ENV{VULKAN_SDK_INSTALL_DIRECTORY}/Bin"
  HINTS "$ENV{VK_SDK_PATH}/Bin"
  HINTS "$ENV{VULKAN_SDK}/Bin"
	REQUIRED
)

cmake_path(GET Vulkan_INCLUDE_DIR PARENT_PATH VULKAN_SDK_INSTALL_DIRECTORY)
get_filename_component(VULKAN_SDK_VERSION "${VULKAN_SDK_INSTALL_DIRECTORY}" NAME)

if(NOT EXISTS "${VULKAN_SDK_INSTALL_DIRECTORY}")
  message(FATAL_ERROR "Internal error, VULKAN_SDK_INSTALL_DIRECTORY doesn't exist")
endif()

find_program(CTEST_EXE
	NAMES ctest
	REQUIRED
)

set(NBL_DOCKER_NSC_COMPILER_CONFIG_OUTPUT "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}/hlsl.local.properties.cmake")

set(NBL_DOCKER_CT_NSC_VOLUME_TARGET "C:\\\\nsc\\\\install")
string(GENEX_STRIP "${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}" NBL_RELATIVE_ENTRY)
set(NSC_RELEASE_BUILD_INFO "${NBL_NSC_PREINSTALL_DIRECTORY}/${NBL_RELATIVE_ENTRY}/${NBL_NSC_BUILD_INFO_FILENAME}")
set(NSC_RELWITHDEBINFO_BUILD_INFO "${NBL_NSC_PREINSTALL_DIRECTORY}/relwithdebinfo/${NBL_RELATIVE_ENTRY}/${NBL_NSC_BUILD_INFO_FILENAME}")
set(NSC_DEBUG_BUILD_INFO "${NBL_NSC_PREINSTALL_DIRECTORY}/debug/${NBL_RELATIVE_ENTRY}/${NBL_NSC_BUILD_INFO_FILENAME}")
cmake_path(NATIVE_PATH NSC_RELEASE_BUILD_INFO NORMALIZE NSC_RELEASE_BUILD_INFO)
cmake_path(NATIVE_PATH NSC_RELWITHDEBINFO_BUILD_INFO NORMALIZE NSC_RELWITHDEBINFO_BUILD_INFO)
cmake_path(NATIVE_PATH NSC_DEBUG_BUILD_INFO NORMALIZE NSC_DEBUG_BUILD_INFO)

set(NBL_INSTALL_DIRECTORY "${NBL_DOCKER_CT_NSC_VOLUME_TARGET}")
cmake_path(NATIVE_PATH NBL_DOCKER_CT_NSC_VOLUME_TARGET NORMALIZE NBL_DOCKER_CT_NSC_VOLUME_TARGET)

set(NBL_BUILD_INFO_POSTPROCESS_COMMAND
  "${CMAKE_COMMAND}"
  "-DNBL_EXECUTABLE_PATH=${NBL_NSC_PREINSTALL_TARGET_EXE_FILEPATH}"
  "-DNBL_BUILD_INFO=${NBL_NSC_PREINSTALL_TARGET_BUILD_INFO}"
  "-DNBL_OUTPUT_FILE=${NBL_NSC_PREINSTALL_TARGET_BUILD_INFO}"
  "-DNBL_OUTPUT_EXE_OVERRIDE=$<PATH:NORMAL_PATH,${NBL_DOCKER_CT_NSC_VOLUME_TARGET}/${NBL_PACKAGE_RUNTIME_EXE_DIR_PATH}/${NBL_NSC_PREINSTALL_TARGET_EXE_FILENAME}>" # as in CT, it's *not* host exe location!
  -P "${NBL_ROOT_PATH}/cmake/scripts/nbl/nablaBuildInfo.cmake"
)

cmake_path(GET SPIRV_DIS_EXE PARENT_PATH VULKAN_SDK_BIN_DIRECTORY)
cmake_path(NATIVE_PATH VULKAN_SDK_BIN_DIRECTORY NORMALIZE VULKAN_SDK_BIN_DIRECTORY)
cmake_path(GET SPIRV_DIS_EXE FILENAME SPIRV_DIS_EXE)
set(CT_SPIRV_DIS_EXE "C:\\vulkan\\${VULKAN_SDK_VERSION}\\bin\\${SPIRV_DIS_EXE}")
cmake_path(NATIVE_PATH CT_SPIRV_DIS_EXE NORMALIZE CT_SPIRV_DIS_EXE)

set(NBL_CE_GENERATE_CONFIG_COMMAND
  "${CMAKE_COMMAND}"
  "-DSPIRV_DIS_EXE=${CT_SPIRV_DIS_EXE}"
  "-DNSC_RELEASE_BUILD_INFO=${NSC_RELEASE_BUILD_INFO}"
  "-DNSC_RELWITHDEBINFO_BUILD_INFO=${NSC_RELWITHDEBINFO_BUILD_INFO}"
  "-DNSC_DEBUG_BUILD_INFO=${NSC_DEBUG_BUILD_INFO}"
  "-DOUTPUT_CONFIG_FILE=${NBL_DOCKER_NSC_COMPILER_CONFIG_OUTPUT}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/ce-generate-config.cmake"
)

set(NBL_DOCKER_CE_DOCKER_CTX "${NBL_ROOT_PATH}/docker/compiler-explorer")
set(NBL_DOCKER_CE_DOCKERFILE_BASE "${NBL_DOCKER_CE_DOCKER_CTX}/Dockerfile")
set(NBL_DOCKER_CE_COMPOSE_BASE "${NBL_DOCKER_CE_DOCKER_CTX}/compose.yml")
cmake_path(NATIVE_PATH NBL_DOCKER_CE_COMPOSE_BASE NORMALIZE NBL_DOCKER_CE_COMPOSE_BASE)
set(NBL_DOCKER_CE_COMPOSE_TARGET "${GODBOLT_BINARY_DIRECTORY}/.dev-compose.yml")

include(InstallRequiredSystemLibraries)

string(REPLACE "v" "VC" TARGET_DCRT ${CMAKE_VS_PLATFORM_TOOLSET})
set(DEBUG_CRT_RELATIVE debug_nonredist/x64/Microsoft.${TARGET_DCRT}.DebugCRT)
set(DEBUG_CRT_DIRECTORY_SOURCE "${MSVC_REDIST_DIR}/${DEBUG_CRT_RELATIVE}")
cmake_path(NATIVE_PATH MSVC_REDIST_DIR NORMALIZE NBL_REDIST_DIR)

if(NOT EXISTS "${DEBUG_CRT_DIRECTORY_SOURCE}")
  message(FATAL_ERROR "DEBUG_CRT_DIRECTORY_SOURCE = \"${DEBUG_CRT_DIRECTORY_SOURCE}\" doesn't exist!")
endif()

set(DEBUG_CRT_DIRECTORY_TARGET "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}/.nonredist")
file(MAKE_DIRECTORY "${DEBUG_CRT_DIRECTORY_TARGET}")
file(GLOB CRT_FILES "${DEBUG_CRT_DIRECTORY_SOURCE}/*")

find_file(UCRTBASED_DLL_PATH
    NAMES ucrtbased.dll
    REQUIRED
)

# TODO: (***) ---> THIS GOES TO <NBL_ROOT_DIR>/docker to CMakeLists.txt file!

set(BASE_IMAGE mcr.microsoft.com/windows/servercore:ltsc2022-amd64) # NOTE: HARDCODED CURRENTLY

string(TOLOWER "dr.devsh.eu/nabla/cmake-host-dev-env/${CMAKE_SYSTEM_NAME}/package/vulkan:latest" DOCKER_VULKAN_TAG)
string(TOLOWER "dr.devsh.eu/nabla/cmake-host-dev-env/${CMAKE_SYSTEM_NAME}/toolset/redist/${CMAKE_CXX_COMPILER_ID}/crt:latest" DOCKER_CRT_TAG)
string(TOLOWER "dr.devsh.eu/nabla/cmake-host-dev-env/${CMAKE_SYSTEM_NAME}/build/${CMAKE_CXX_COMPILER_ID}/devel-compiler-explorer-nsc:latest" DOCKER_DEVEL_TAG)

cmake_path(NATIVE_PATH MSVC_REDIST_DIR NORMALIZE TOOLSET_REDIST_PATH)
get_filename_component(REDIST_CRT_TOOLSET_VERSION "${TOOLSET_REDIST_PATH}" NAME)

function(GEN_DOCKER_CONTENT _CTX_ _OUTPUT_DIRECTORY_ _EXTRA_DOCKERFILE_CONTENT_ _DOCKER_IGNORE_CONTENT_ _S_NAME_ _CT_NAME_ _IMAGE_NAME_ _WITH_BUILD_) 

set(_OUTPUT_D_PATH_ "${_OUTPUT_DIRECTORY_}/Dockerfile")
set(_OUTPUT_C_PATH_ "${_OUTPUT_DIRECTORY_}/compose.yml")

string(CONFIGURE "${_EXTRA_DOCKERFILE_CONTENT_}" _EXTRA_DOCKERFILE_CONTENT_EVAL_ @ONLY)
string(CONFIGURE "${_DOCKER_IGNORE_CONTENT_}" _DOCKER_IGNORE_CONTENT_EVAL_ @ONLY)

unset(DOCKER_CONTENT)
string(APPEND DOCKER_CONTENT
[=[
# escape=`

ARG BASE_IMAGE=@BASE_IMAGE@
FROM ${BASE_IMAGE}
SHELL ["cmd", "/S", "/C"]
@_EXTRA_DOCKERFILE_CONTENT_EVAL_@
]=]
)

string(CONFIGURE "${DOCKER_CONTENT}" DOCKER_CONTENT @ONLY)
file(WRITE "${_OUTPUT_D_PATH_}" "${DOCKER_CONTENT}")

set(_CTX_TARGET_ "${_OUTPUT_DIRECTORY_}/.ctx")

if("${_CTX_}" STREQUAL "")

else()
  if(NOT EXISTS "${_CTX_}")
    message(FATAL_ERROR "Invalid source context directory doesn't exist! _CTX_: \"${_CTX_}\"")
  endif()

  file(COPY "${_CTX_}" DESTINATION "${_CTX_TARGET_}")
endif()

set(_OUTPUT_I_PATH_ "${_CTX_TARGET_}/.dockerignore")

unset(COMPOSE_CONTENT)
string(APPEND COMPOSE_CONTENT
[=[
services:
  @_S_NAME_@:
    build:
      context: ./.ctx
      dockerfile: "@_OUTPUT_D_PATH_@"
    image: @_IMAGE_NAME_@
    container_name: @_CT_NAME_@
    networks:
      docker_default:

networks:
  docker_default:
    external: true
]=]
)

string(CONFIGURE "${COMPOSE_CONTENT}" COMPOSE_CONTENT @ONLY)
file(WRITE "${_OUTPUT_C_PATH_}" "${COMPOSE_CONTENT}")
file(WRITE "${_OUTPUT_I_PATH_}" "${_DOCKER_IGNORE_CONTENT_EVAL_}")

if(_WITH_BUILD_)
  execute_process(COMMAND "${DOCKER_EXE}" compose -f "${_OUTPUT_C_PATH_}" build)
endif()
endfunction()

# Vulkan
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docker/vulkan")
set(CT_VULKAN_TARGET vulkan)
GEN_DOCKER_CONTENT("${VULKAN_SDK_INSTALL_DIRECTORY}" "${OUTPUT_DIRECTORY}"
[=[
COPY ./ "@CT_VULKAN_TARGET@"

ENV VULKAN_SDK="C:/@CT_VULKAN_TARGET@"
ENV VULKAN_SDK_VERSION="@VULKAN_SDK_VERSION@"
LABEL VULKAN_SDK="C:/@CT_VULKAN_TARGET@"
LABEL VULKAN_SDK_VERSION="@VULKAN_SDK_VERSION@"
]=]
[=[
*
!@VULKAN_SDK_VERSION@/Bin/*.dll
!@VULKAN_SDK_VERSION@/Bin/*spirv*.exe
]=]
nabla-dev-env-vulkan 
nabla.dev.env.vulkan
${DOCKER_VULKAN_TAG}
ON
)

# CRT
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docker/crt")
set(CT_TOOLSET_REDIST_TARGET toolset_redist)
make_directory("${OUTPUT_DIRECTORY}/.ctx")
file(COPY "${UCRTBASED_DLL_PATH}" DESTINATION "${OUTPUT_DIRECTORY}/.ctx")
GEN_DOCKER_CONTENT("${TOOLSET_REDIST_PATH}" "${OUTPUT_DIRECTORY}"
[=[
COPY ./ "/@CT_TOOLSET_REDIST_TARGET@"

ENV REDIST_CRT_TOOLSET_VERSION="@REDIST_CRT_TOOLSET_VERSION@"
ENV TOOLSET_REDIST_PATH="C:/@CT_TOOLSET_REDIST_TARGET@"
LABEL REDIST_CRT_TOOLSET_VERSION="@REDIST_CRT_TOOLSET_VERSION@"
LABEL TOOLSET_REDIST_PATH="C:/@CT_TOOLSET_REDIST_TARGET@"
]=]
[=[
*
!ucrtbased.dll
!@REDIST_CRT_TOOLSET_VERSION@/vc_redist.x64.exe
!@REDIST_CRT_TOOLSET_VERSION@/@DEBUG_CRT_RELATIVE@/*.dll
]=]
nabla-dev-env-crt 
nabla.dev.env.crt
${DOCKER_CRT_TAG}
ON
)

# Devel, combined
set(BASE_IMAGE dr.devsh.eu/compiler-explorer/windows)

# NOTE to self: could be all done with single docker file & compose file but buildkit works bad with windows driver, yet need to wait for stuff to be implemented
set(DEVEL_CTX "${CMAKE_CURRENT_BINARY_DIR}/docker/devel")
set(CT_REDIST_DIR "${CT_TOOLSET_REDIST_TARGET}/${REDIST_CRT_TOOLSET_VERSION}")
set(CT_NONREDIST_CTR_DIR "${CT_REDIST_DIR}/${DEBUG_CRT_RELATIVE}")
cmake_path(NATIVE_PATH CT_REDIST_DIR NORMALIZE CT_REDIST_DIR)
cmake_path(NATIVE_PATH CT_NONREDIST_CTR_DIR NORMALIZE CT_NONREDIST_CTR_DIR)
set(DEVEL_DOCKERFILE "${DEVEL_CTX}/Dockerfile")

GEN_DOCKER_CONTENT("" "${DEVEL_CTX}"
[=[

COPY --link --from=@DOCKER_VULKAN_TAG@ /@CT_VULKAN_TARGET@ /@CT_VULKAN_TARGET@
COPY --link --from=@DOCKER_CRT_TAG@ /@CT_TOOLSET_REDIST_TARGET@ /@CT_TOOLSET_REDIST_TARGET@

# TODO
# RUN .\@CT_REDIST_DIR@\vc_redist.x64.exe /quiet /install 
RUN xcopy .\@CT_NONREDIST_CTR_DIR@\*.dll %SystemRoot%\System32 /Y
RUN xcopy .\@CT_TOOLSET_REDIST_TARGET@\ucrtbased.dll %SystemRoot%\System32 /Y

]=]
[=[

]=]
nabla-dev-env-nsc 
nabla.dev.env.nsc
${DOCKER_DEVEL_TAG}
OFF
)

# <---(***)

set(NABLA_DEV_ENV_CT_NAME dev.nabla.env.${CMAKE_SYSTEM_NAME}.${CMAKE_CXX_COMPILER_ID}.base)
string(TOLOWER "${NABLA_DEV_ENV_CT_NAME}" NABLA_DEV_ENV_CT_NAME)

set(COMPOSE_NSC_DEV_SERVICE compiler-explorer-nsc-dev)
string(TOLOWER "dr.devsh.eu/nabla/cmake-host-dev-env/${CMAKE_SYSTEM_NAME}/build/${CMAKE_CXX_COMPILER_ID}/compiler-explorer-nsc:latest" COMPOSE_NSC_DEV_IMAGE)
string(TOLOWER "dr.devsh.eu/compiler-explorer/production/windows/nsc/orphan-production-test:latest" COMPOSE_NSC_ORPHAN_PRODUCTION_TEST_IMAGE)
string(TOLOWER "dr.devsh.eu/compiler-explorer/production/windows/nsc/orphan-prodution-cache:latest" COMPOSE_NSC_PRODUCTION_CACHE_IMAGE)
string(TOLOWER "dr.devsh.eu/compiler-explorer/production/windows/nsc:latest" COMPOSE_NSC_PRODUCTION_IMAGE)

string(APPEND COMPOSE_CONTENT
[=[
services:
  @COMPOSE_NSC_DEV_SERVICE@:
    container_name: dev.ce.nsc.dev
    extends:
        file: @NBL_DOCKER_CE_COMPOSE_BASE@
        service: compiler-explorer
    build:
      context: ./.ctx
      dockerfile: @DEVEL_DOCKERFILE@
    image: @COMPOSE_NSC_DEV_IMAGE@
    environment:
      NBL_INSTALL_DIRECTORY: "@NBL_INSTALL_DIRECTORY@"
      NBL_EXPLICIT_MODULE_LOAD_LOG: "ON"
    entrypoint:
      - "cmd"
      - "/c"
      - >
        copy C:\\nsc\\install\\hlsl.local.properties.cmake %GIT_GODBOLT_REPOSITORY_PATH%\\etc\\config\\hlsl.local.properties
        && npm --prefix %GIT_GODBOLT_REPOSITORY_PATH% run dev -- --language hlsl
    volumes:
      - type: bind
        source: .\install
        target: @NBL_DOCKER_CT_NSC_VOLUME_TARGET@
        read_only: true

networks:
  docker_default:
    external: true
]=]
)

string(CONFIGURE "${COMPOSE_CONTENT}" COMPOSE_CONTENT @ONLY)
file(WRITE "${NBL_DOCKER_CE_COMPOSE_TARGET}" "${COMPOSE_CONTENT}")
make_directory("${GODBOLT_BINARY_DIRECTORY}/.ctx")

function(_PROMOTE_PROCESS_ISOLATION_ KERNEL BASES VAR)
    set(${VAR} True)
    set(ix 0)
    list(LENGTH BASES LEN)

    while(ix LESS ${LEN})
        list(GET BASES ${ix} BASE)

        execute_process(COMMAND "${DOCKER_EXE}" inspect --format={{.OsVersion}} ${BASE} RESULT_VARIABLE EXIT_LEVEL OUTPUT_VARIABLE TARGET_KERNEL OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(${EXIT_LEVEL} EQUAL 0)
            if(${KERNEL} VERSION_LESS ${TARGET_KERNEL})
                set(${VAR} False PARENT_SCOPE)
                message(STATUS "While inspecting ${BASE} - host Kernel ${KERNEL} too low to use container process isolation (target ${TARGET_KERNEL}), falling back to HyperV. Please update your host OS.")
                return()
            endif()
            math(EXPR ix "${ix} + 1")
        else()
            message(STATUS "Docker image ${BASE} not found locally, pulling...")
            execute_process(COMMAND "${DOCKER_EXE}" pull ${BASE})
        endif()
    endwhile()

    set(${VAR} ${${VAR}} PARENT_SCOPE)
endfunction()

execute_process(COMMAND cmd /C ver OUTPUT_VARIABLE PIPE OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+" HOST_KERNEL "${PIPE}")

set(BASES
  mcr.microsoft.com/windows/nanoserver:ltsc2022
  mcr.microsoft.com/powershell:lts-nanoserver-ltsc2022
)

_PROMOTE_PROCESS_ISOLATION_("${HOST_KERNEL}" "${BASES}" PROMOTE_TO_PROCESS)

function(_BUILD_IMAGE_ DOCKERFILE CTX TAG)
  set(CMD "${DOCKER_EXE}" build)
  if(PROMOTE_TO_PROCESS)
    list(APPEND CMD --isolation "process")
  endif()
  list(APPEND CMD -t ${TAG} -f "${DOCKERFILE}" .)

  execute_process(COMMAND ${CMD} WORKING_DIRECTORY "${CTX}") 
endfunction()

_BUILD_IMAGE_("${NBL_DOCKER_CE_DOCKERFILE_BASE}" "${NBL_DOCKER_CE_DOCKER_CTX}" godbolt/base/windows)
_BUILD_IMAGE_("${DEVEL_DOCKERFILE}" "${DEVEL_CTX}" godbolt/devel/windows)

message(FATAL_ERROR "STOP TEST, PROMOTE_TO_PROCESS = ${PROMOTE_TO_PROCESS}")

string(APPEND BAT_PRODUCTION_INSTALL
[=[
@echo off
setlocal

set BASE_PATH=C:\

xcopy "%BASE_PATH%target" "%BASE_PATH%nsc\install" /s /e /h /i /y /f
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to copy C:\target to C:\nsc\install
    exit /b %ERRORLEVEL%
)

if "%GIT_GODBOLT_REPOSITORY_PATH%"=="" (
    echo [ERROR] Environment variable GIT_GODBOLT_REPOSITORY_PATH is not set!
    exit /b 1
)

copy "%BASE_PATH%nsc\install\hlsl.local.properties.cmake" "%GIT_GODBOLT_REPOSITORY_PATH%\etc\config\hlsl.local.properties"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to copy HLSL properties file
    exit /b %ERRORLEVEL%
)

echo [SUCCESS] All production files copied successfully.
exit /b 0
]=]
)

string(CONFIGURE "${BAT_PRODUCTION_INSTALL}" BAT_PRODUCTION_INSTALL @ONLY)
file(WRITE "${NBL_DOCKER_CT_NSC_INSTALL_BAT}" "${BAT_PRODUCTION_INSTALL}")

set(NBL_CE_URL http://localhost:80)
set(NBL_CE_HEALTHY_CHECK_PY "${NBL_ROOT_PATH}/docker/compiler-explorer/ce_healthy_check.py")
set(NBL_CE_ENDPOINT_PY "${NBL_ROOT_PATH}/docker/compiler-explorer/endpoint.py")
set(NBL_NSC_BASIC_HLSL_JPAYLOAD "${CMAKE_CURRENT_SOURCE_DIR}/docker/godbolt/hlsl-basic-compile-payload.json")

add_custom_target(run-compiler-explorer
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Performing Pre-Test..."
    COMMAND "${CTEST_EXE}" -C $<CONFIG> --stop-on-failure
    COMMAND ${NBL_BUILD_INFO_POSTPROCESS_COMMAND}
    COMMAND "${DOCKER_EXE}" compose -f "${NBL_DOCKER_CE_COMPOSE_TARGET}" stop ${COMPOSE_NSC_DEV_SERVICE}
    COMMAND ${NBL_CE_GENERATE_CONFIG_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --green "OK! Performing executables hot-swap..."
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${NBL_NSC_PREINSTALL_DIRECTORY}" "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}"
    COMMAND "${DOCKER_EXE}" compose -f "${NBL_DOCKER_CE_COMPOSE_TARGET}" up -d ${COMPOSE_NSC_DEV_SERVICE}
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Checking health of Compiler Explorer service..."
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_HEALTHY_CHECK_PY}" --url "${NBL_CE_URL}" --interval 10 --ticks 25
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Compiler Explorer is running, type \"localhost\" in your browser!"
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Post-Checking if NSC is able to compile basic shader file..."
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_ENDPOINT_PY}" --url "${NBL_CE_URL}" --endpoint /api/compiler/nsc_$<LOWER_CASE:$<CONFIG>>_upstream/compile --method POST --json "${NBL_NSC_BASIC_HLSL_JPAYLOAD}"
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "OK! NSC is healthy."
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    VERBATIM 
    USES_TERMINAL
)

add_custom_target(is-compiler-explorer-running
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_HEALTHY_CHECK_PY}" --url "${NBL_CE_URL}" --ticks 1
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_ENDPOINT_PY}" --url "${NBL_CE_URL}" --endpoint /api/compilers
    VERBATIM
    USES_TERMINAL
)

# Production NSC image
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docker/nsc-production")
set(BASE_IMAGE "${COMPOSE_NSC_ORPHAN_PRODUCTION_TEST_IMAGE}")
set(NBL_DOCKER_TMP_PRODUCTION_TARGET "C:\\target")
GEN_DOCKER_CONTENT("" "${OUTPUT_DIRECTORY}"
[=[
LABEL maintainer="Arkadiusz Lachowicz <a.lachowicz@devsh.eu>" `
  org.opencontainers.image.authors="Arkadiusz Lachowicz <a.lachowicz@devsh.eu>" `
  org.opencontainers.image.title="Compiler Explorer with Nabla Shader Compilers in Docker" `
  org.opencontainers.image.description="Docker image to run Compiler Explorer instance with Nabla Shader Compilers" `
  org.opencontainers.image.url="https://github.com/Devsh-Graphics-Programming/Nabla" `
  org.opencontainers.image.source="https://github.com/Devsh-Graphics-Programming/Nabla" `
  org.opencontainers.image.documentation="https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/tools/nsc/docker"

ENTRYPOINT ["powershell.exe", "-ExecutionPolicy", "Bypass", "-Command", "npm", "--prefix", "$env:GIT_GODBOLT_REPOSITORY_PATH", "start", "--", "--language", "hlsl"]
]=]
[=[

]=]
nsc-ce-production-cache-webpack
nsc.ce.production.cache.webpack
${COMPOSE_NSC_PRODUCTION_CACHE_IMAGE}
OFF
)

set(NBL_CE_URL http://localhost:6969)

add_custom_target(create-production-compiler-explorer
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Removing any remaining pre-test orphan containers..." 
    COMMAND "${DOCKER_EXE}" rm -f production-ce-orphan-run-test || "${CMAKE_COMMAND}" -E true
    COMMAND "${DOCKER_EXE}" rm -f production-ce-orphan-cache-webpack || "${CMAKE_COMMAND}" -E true
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Creating pre-test production image..."
    COMMAND "${DOCKER_EXE}" run -dit -v "${NBL_DOCKER_CT_NSC_VOLUME_SOURCE}:${NBL_DOCKER_TMP_PRODUCTION_TARGET}" --name production-ce-orphan-run-test --entrypoint "cmd" "${COMPOSE_NSC_DEV_IMAGE}"
    COMMAND "${DOCKER_EXE}" exec production-ce-orphan-run-test "${NBL_DOCKER_TMP_PRODUCTION_TARGET}\\${NBL_DOCKER_INSTALL_BAT_FILENAME}"
    COMMAND "${DOCKER_EXE}" stop production-ce-orphan-run-test
    COMMAND "${DOCKER_EXE}" commit -m "Copy NSC install redists" production-ce-orphan-run-test "${COMPOSE_NSC_ORPHAN_PRODUCTION_TEST_IMAGE}"
    COMMAND "${DOCKER_EXE}" compose build
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Running pre-test production image, caching webpack & running final checks..."
    COMMAND "${DOCKER_EXE}" run -dit -p 6969:10240 --name production-ce-orphan-cache-webpack "${COMPOSE_NSC_PRODUCTION_CACHE_IMAGE}"
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_HEALTHY_CHECK_PY}" --url "${NBL_CE_URL}" --interval 10 --ticks 35
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_ENDPOINT_PY}" --url "${NBL_CE_URL}" --endpoint /api/compilers --disable-cookies --timeout 69
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_ENDPOINT_PY}" --url "${NBL_CE_URL}" --endpoint /api/compiler/nsc_release_upstream/compile --method POST --json "${NBL_NSC_BASIC_HLSL_JPAYLOAD}" --disable-cookies --timeout 69
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_ENDPOINT_PY}" --url "${NBL_CE_URL}" --endpoint /api/compiler/nsc_relwithdebinfo_upstream/compile --method POST --json "${NBL_NSC_BASIC_HLSL_JPAYLOAD}" --disable-cookies --timeout 69
    COMMAND "${_Python3_EXECUTABLE}" "${NBL_CE_ENDPOINT_PY}" --url "${NBL_CE_URL}" --endpoint /api/compiler/nsc_debug_upstream/compile --method POST --json "${NBL_NSC_BASIC_HLSL_JPAYLOAD}" --disable-cookies --timeout 69
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --blue "Passed all tests! Creating final production image..."
    COMMAND "${DOCKER_EXE}" stop production-ce-orphan-cache-webpack
    COMMAND "${DOCKER_EXE}" commit -m "Perform tests, cache webpack build" production-ce-orphan-cache-webpack "${COMPOSE_NSC_PRODUCTION_IMAGE}"
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --green "Created final `${COMPOSE_NSC_PRODUCTION_IMAGE}` production image!"
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --green "To run the production image, execute: 'docker run -p 80:10240 ${COMPOSE_NSC_PRODUCTION_IMAGE}',"
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --green "'docker run -p 80:10240 ${COMPOSE_NSC_PRODUCTION_IMAGE}'."
    COMMAND "${CMAKE_COMMAND}" -E cmake_echo_color --green "The production image can be pushed safely to the public registry."
    WORKING_DIRECTORY "${OUTPUT_DIRECTORY}"
    VERBATIM
    USES_TERMINAL
)

add_dependencies(run-compiler-explorer nsc)
set_target_properties(run-compiler-explorer PROPERTIES FOLDER "Godbolt")
set_target_properties(is-compiler-explorer-running PROPERTIES FOLDER "Godbolt")
set_target_properties(create-production-compiler-explorer PROPERTIES FOLDER "Godbolt")

endif()