cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED SMOKE_EXE)
    message(FATAL_ERROR "SMOKE_EXE is required")
endif()
if(NOT DEFINED OVERRIDE_DIR)
    message(FATAL_ERROR "OVERRIDE_DIR is required")
endif()
if(NOT EXISTS "${SMOKE_EXE}")
    message(FATAL_ERROR "SMOKE_EXE not found: ${SMOKE_EXE}")
endif()

function(find_module_file OUT_VAR DIR_PATH MODULE_BASENAME REQUIRED_FLAG)
    if(NOT EXISTS "${DIR_PATH}")
        if(REQUIRED_FLAG)
            message(FATAL_ERROR "Runtime directory not found: ${DIR_PATH}")
        endif()
        set(${OUT_VAR} "" PARENT_SCOPE)
        return()
    endif()

    set(_candidates "${MODULE_BASENAME}")
    if(WIN32)
        list(APPEND _candidates "${MODULE_BASENAME}.dll")
    elseif(APPLE)
        list(APPEND _candidates "${MODULE_BASENAME}.dylib" "lib${MODULE_BASENAME}.dylib")
    else()
        list(APPEND _candidates "${MODULE_BASENAME}.so" "lib${MODULE_BASENAME}.so")
    endif()

    foreach(_name IN LISTS _candidates)
        set(_candidate_path "${DIR_PATH}/${_name}")
        if(EXISTS "${_candidate_path}" AND NOT IS_DIRECTORY "${_candidate_path}")
            set(${OUT_VAR} "${_candidate_path}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    if(NOT WIN32 AND NOT APPLE)
        file(GLOB _versioned_candidates
            "${DIR_PATH}/${MODULE_BASENAME}.so.*"
            "${DIR_PATH}/lib${MODULE_BASENAME}.so.*"
        )
        list(LENGTH _versioned_candidates _versioned_count)
        if(_versioned_count GREATER 0)
            list(GET _versioned_candidates 0 _versioned_first)
            set(${OUT_VAR} "${_versioned_first}" PARENT_SCOPE)
            return()
        endif()
    endif()

    if(REQUIRED_FLAG)
        message(FATAL_ERROR "Could not find module ${MODULE_BASENAME} in ${DIR_PATH}")
    endif()

    set(${OUT_VAR} "" PARENT_SCOPE)
endfunction()

file(MAKE_DIRECTORY "${OVERRIDE_DIR}")

if(DEFINED DXC_RUNTIME_DIR AND DEFINED DXC_MODULE_BASENAME AND NOT "${DXC_MODULE_BASENAME}" STREQUAL "")
    find_module_file(_dxc_module_file "${DXC_RUNTIME_DIR}" "${DXC_MODULE_BASENAME}" TRUE)
else()
    message(FATAL_ERROR "DXC_RUNTIME_DIR and DXC_MODULE_BASENAME are required")
endif()

if(DEFINED NABLA_RUNTIME_DIR AND EXISTS "${NABLA_RUNTIME_DIR}")
    if(DEFINED NABLA_MODULE_BASENAME AND NOT "${NABLA_MODULE_BASENAME}" STREQUAL "")
        find_module_file(_nabla_module_file "${NABLA_RUNTIME_DIR}" "${NABLA_MODULE_BASENAME}" FALSE)
        set(_nabla_modules "${_nabla_module_file}")
    else()
        if(WIN32)
            file(GLOB _nabla_modules "${NABLA_RUNTIME_DIR}/Nabla*.dll")
        elseif(APPLE)
            file(GLOB _nabla_modules "${NABLA_RUNTIME_DIR}/Nabla*.dylib" "${NABLA_RUNTIME_DIR}/libNabla*.dylib")
        else()
            file(GLOB _nabla_modules "${NABLA_RUNTIME_DIR}/Nabla*.so" "${NABLA_RUNTIME_DIR}/Nabla*.so.*" "${NABLA_RUNTIME_DIR}/libNabla*.so" "${NABLA_RUNTIME_DIR}/libNabla*.so.*")
        endif()
    endif()

    foreach(_nabla_module IN LISTS _nabla_modules)
        if(NOT "${_nabla_module}" STREQUAL "")
            execute_process(
                COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_nabla_module}" "${OVERRIDE_DIR}"
                RESULT_VARIABLE _copy_nabla_rv
            )
            if(NOT _copy_nabla_rv EQUAL 0)
                message(FATAL_ERROR "Failed to copy Nabla module from ${_nabla_module} to ${OVERRIDE_DIR}")
            endif()
        endif()
    endforeach()
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_dxc_module_file}" "${OVERRIDE_DIR}"
    RESULT_VARIABLE _copy_dxc_rv
)
if(NOT _copy_dxc_rv EQUAL 0)
    message(FATAL_ERROR "Failed to copy DXC module from ${_dxc_module_file} to ${OVERRIDE_DIR}")
endif()

execute_process(
    COMMAND "${SMOKE_EXE}"
    RESULT_VARIABLE _smoke_rv
)
if(NOT _smoke_rv EQUAL 0)
    message(FATAL_ERROR "smoke_override failed with exit code ${_smoke_rv}")
endif()
