include(common)

set(NSC_JSON_REPORT ON)
set(NSC_SHADER_CACHE ON)
set(NSC_PREPROCESS_CACHE ON)

if(NOT Python3_EXECUTABLE)
	find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

set(NBL_NSC_CACHE_TEST_SEED "0" CACHE STRING "Seed for NSC cache layer tests (0 = deterministic)")
set(NBL_NSC_CACHE_TEST_ITERATIONS "5" CACHE STRING "Iterations for NSC cache layer stress test")
set(NBL_NSC_CACHE_TEST_PARALLEL_JOBS "3" CACHE STRING "Parallel jobs for NSC cache layer test")
set(NBL_NSC_CACHE_PREAMBLE_BUDGET_MS "0" CACHE STRING "Optional max total_with_output_ms for preamble hit time test (0 disables check)")

set(NBL_NSC_CACHE_LAYER_ROOT "${CMAKE_CURRENT_BINARY_DIR}/cache_layers")
set(NBL_NSC_CACHE_LAYER_SRC "${NBL_NSC_CACHE_LAYER_ROOT}/src")
file(MAKE_DIRECTORY "${NBL_NSC_CACHE_LAYER_SRC}")

set(NBL_NSC_CACHE_PROXY "${NBL_NSC_CACHE_LAYER_SRC}/proxy.hlsl")
file(WRITE "${NBL_NSC_CACHE_PROXY}" [=[
#ifndef NBL_NSC_CACHE_TEST_PROXY_HLSL
#define NBL_NSC_CACHE_TEST_PROXY_HLSL
// NBL_NSC_CACHE_TEST_DEFINES_BEGIN
// NBL_NSC_CACHE_TEST_DEFINES_END
// NBL_NSC_CACHE_TEST_INCLUDES_BEGIN
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>
#include <nbl/builtin/hlsl/cpp_compat/vector.hlsl>
// NBL_NSC_CACHE_TEST_INCLUDES_END
#endif
]=])

function(nbl_nsc_write_input _path)
	file(WRITE "${_path}" [=[
#include "proxy.hlsl"

[numthreads(1,1,1)]
[shader("compute")]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint sink = 1u;
    if (tid.x == 0u && sink == 0u)
        return;
}
]=])
endfunction()

set(NBL_NSC_CACHE_INPUT "${NBL_NSC_CACHE_LAYER_SRC}/cache_layers_input.hlsl")
nbl_nsc_write_input("${NBL_NSC_CACHE_INPUT}")
set(NBL_NSC_SHADER_INPUT "${NBL_NSC_CACHE_INPUT}")
set(NBL_NSC_PREPROCESS_INPUT "${NBL_NSC_CACHE_INPUT}")
set(NBL_NSC_PREAMBLE_INPUT "${NBL_NSC_CACHE_INPUT}")

function(nbl_nsc_add_cache_target _name _input _binary_dir _output_var _export_prefix)
	set(JSON_TEMPLATE [=[[
  {
    "INPUT": "@INPUT@",
    "KEY": "@KEY@",
    "COMPILE_OPTIONS": ["-T", "cs_6_7"],
    "CAPS": []
  }
]]]=])
	set(INPUT "${_input}")
	set(KEY "${_name}")
	string(CONFIGURE "${JSON_TEMPLATE}" JSON @ONLY)

	NBL_CREATE_NSC_COMPILE_RULES(
		TARGET ${_name}_spirv
		BINARY_DIR "${_binary_dir}"
		MOUNT_POINT_DEFINE NBL_NSC_CACHE_LAYERS_MOUNT
		COMMON_OPTIONS -I "${NBL_NSC_CACHE_LAYER_SRC}"
		OUTPUT_VAR ${_output_var}
		EXPORT_RULES ${_export_prefix}
		DISABLE_CUSTOM_COMMANDS
		INCLUDE nbl/nsc/tests/cache_layers/keys.hpp
		NAMESPACE nbl::nsc::tests::cache_layers
		INPUTS ${JSON}
		DISCARD_DEFAULT_GLOB
	)

	if(DEFINED ${_output_var})
		set(${_output_var} ${${_output_var}} PARENT_SCOPE)
	endif()
	if(_export_prefix AND DEFINED ${_export_prefix}_COUNT)
		set(${_export_prefix}_COUNT "${${_export_prefix}_COUNT}" PARENT_SCOPE)
		if(${_export_prefix}_COUNT GREATER 0)
			math(EXPR _export_last "${${_export_prefix}_COUNT} - 1")
			foreach(_export_idx RANGE 0 ${_export_last})
				set(${_export_prefix}_COMMAND_${_export_idx} ${${_export_prefix}_COMMAND_${_export_idx}} PARENT_SCOPE)
				set(${_export_prefix}_OUTPUT_${_export_idx} "${${_export_prefix}_OUTPUT_${_export_idx}}" PARENT_SCOPE)
				set(${_export_prefix}_LOG_${_export_idx} "${${_export_prefix}_LOG_${_export_idx}}" PARENT_SCOPE)
				set(${_export_prefix}_DEPFILE_${_export_idx} "${${_export_prefix}_DEPFILE_${_export_idx}}" PARENT_SCOPE)
				set(${_export_prefix}_REPORT_${_export_idx} "${${_export_prefix}_REPORT_${_export_idx}}" PARENT_SCOPE)
			endforeach()
		endif()
	endif()
endfunction()

set(_BIN_SHADER "${NBL_NSC_CACHE_LAYER_ROOT}/shader_cache")
set(_BIN_PREPROCESS "${NBL_NSC_CACHE_LAYER_ROOT}/preprocess_cache")
set(_BIN_PREAMBLE "${NBL_NSC_CACHE_LAYER_ROOT}/preamble_cache")

set(NSC_PREPROCESS_PREAMBLE ON)
set(_EXPORT_SHADER NSC_CACHE_SHADER)
nbl_nsc_add_cache_target(nsc_cache_shader_hit "${NBL_NSC_SHADER_INPUT}" "${_BIN_SHADER}" KEYS_SHADER_CACHE ${_EXPORT_SHADER})

set(NSC_PREPROCESS_PREAMBLE OFF)
set(_EXPORT_PREPROCESS NSC_CACHE_PREPROCESS)
nbl_nsc_add_cache_target(nsc_cache_preprocess_hit "${NBL_NSC_PREPROCESS_INPUT}" "${_BIN_PREPROCESS}" KEYS_PREPROCESS_CACHE ${_EXPORT_PREPROCESS})

set(NSC_PREPROCESS_PREAMBLE ON)
set(_EXPORT_PREAMBLE NSC_CACHE_PREAMBLE)
nbl_nsc_add_cache_target(nsc_cache_preamble_hit "${NBL_NSC_PREAMBLE_INPUT}" "${_BIN_PREAMBLE}" KEYS_PREAMBLE_CACHE ${_EXPORT_PREAMBLE})

set(NBL_NSC_CACHE_TEST_BASE
	"${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cache_layers_test.py"
)

function(nbl_nsc_make_test_args _out _mode _input _output _report _log _depfile _shader_cache _preprocess_cache _preprocessed _command)
	set(_args
		${NBL_NSC_CACHE_TEST_BASE}
		--mode ${_mode}
		--input "$<PATH:NORMAL_PATH,${_input}>"
		--output "$<PATH:NORMAL_PATH,${_output}>"
		--report "$<PATH:NORMAL_PATH,${_report}>"
		--log "$<PATH:NORMAL_PATH,${_log}>"
		--depfile "$<PATH:NORMAL_PATH,${_depfile}>"
		--shader-cache "$<PATH:NORMAL_PATH,${_shader_cache}>"
		--preprocess-cache "$<PATH:NORMAL_PATH,${_preprocess_cache}>"
		--preprocessed "$<PATH:NORMAL_PATH,${_preprocessed}>"
		--seed "${NBL_NSC_CACHE_TEST_SEED}"
	)
	if(ARGN)
		list(APPEND _args ${ARGN})
	endif()
	list(APPEND _args --command ${_command})
	set(${_out} ${_args} PARENT_SCOPE)
endfunction()

function(nbl_nsc_add_cache_tests _prefix _fixture _cold_args _hit_args)
	add_test(NAME ${_prefix}_COLD_RUN_TEST
		COMMAND ${_cold_args}
		COMMAND_EXPAND_LISTS
	)
	add_test(NAME ${_prefix}_HIT_TEST
		COMMAND ${_hit_args}
		COMMAND_EXPAND_LISTS
	)
	set_tests_properties(${_prefix}_COLD_RUN_TEST PROPERTIES FIXTURES_SETUP ${_fixture} RESOURCE_LOCK nbl_nsc_cache_layers)
	set_tests_properties(${_prefix}_HIT_TEST PROPERTIES FIXTURES_REQUIRED ${_fixture} RESOURCE_LOCK nbl_nsc_cache_layers)
endfunction()

function(nbl_nsc_add_single_test _name _args)
	add_test(NAME ${_name}
		COMMAND ${_args}
		COMMAND_EXPAND_LISTS
	)
	set_tests_properties(${_name} PROPERTIES RESOURCE_LOCK nbl_nsc_cache_layers)
endfunction()

function(nbl_nsc_make_args_from_export _out _mode _input _export_prefix)
	set(_command ${${_export_prefix}_COMMAND_0})
	set(_out_path "${${_export_prefix}_OUTPUT_0}")
	set(_report_path "${${_export_prefix}_REPORT_0}")
	set(_log_path "${${_export_prefix}_LOG_0}")
	set(_depfile_path "${${_export_prefix}_DEPFILE_0}")
	set(_cache_shader "${${_export_prefix}_CACHE_SHADER_0}")
	set(_cache_preprocess "${${_export_prefix}_CACHE_PREPROCESS_0}")
	set(_preprocessed "${${_export_prefix}_PREPROCESSED_0}")

	nbl_nsc_make_test_args(${_out} ${_mode} "${_input}" "${_out_path}" "${_report_path}" "${_log_path}" "${_depfile_path}" "${_cache_shader}" "${_cache_preprocess}" "${_preprocessed}" "${_command}" ${ARGN})
	set(${_out} ${${_out}} PARENT_SCOPE)
endfunction()

function(nbl_nsc_path_with_suffix _out _path _suffix)
	cmake_path(GET _path PARENT_PATH _dir)
	cmake_path(GET _path STEM _stem)
	cmake_path(GET _path EXTENSION _ext)
	set(${_out} "${_dir}/${_stem}${_suffix}${_ext}" PARENT_SCOPE)
endfunction()

function(nbl_nsc_make_no_cache_args _out _input _export_prefix)
	set(_command ${${_export_prefix}_COMMAND_0})
	set(_out_path "${${_export_prefix}_OUTPUT_0}")
	set(_report_path "${${_export_prefix}_REPORT_0}")
	set(_log_path "${${_export_prefix}_LOG_0}")
	set(_depfile_path "${${_export_prefix}_DEPFILE_0}")
	set(_cache_shader "${${_export_prefix}_CACHE_SHADER_0}")
	set(_cache_preprocess "${${_export_prefix}_CACHE_PREPROCESS_0}")
	set(_preprocessed "${${_export_prefix}_PREPROCESSED_0}")

	nbl_nsc_path_with_suffix(_out_no_cache "${_out_path}" ".no_cache")
	nbl_nsc_path_with_suffix(_report_no_cache "${_report_path}" ".no_cache")
	nbl_nsc_path_with_suffix(_log_no_cache "${_log_path}" ".no_cache")
	nbl_nsc_path_with_suffix(_depfile_no_cache "${_depfile_path}" ".no_cache")
	nbl_nsc_path_with_suffix(_preprocessed_no_cache "${_preprocessed}" ".no_cache")

	nbl_nsc_make_test_args(${_out} no_cache_cold "${_input}" "${_out_no_cache}" "${_report_no_cache}" "${_log_no_cache}" "${_depfile_no_cache}" "${_cache_shader}" "${_cache_preprocess}" "${_preprocessed_no_cache}" "${_command}")
	set(${_out} ${${_out}} PARENT_SCOPE)
endfunction()

function(nbl_nsc_register_cache_layer _prefix _fixture _mode_base _input _export_prefix)
	set(_command ${${_export_prefix}_COMMAND_0})
	set(_out "${${_export_prefix}_OUTPUT_0}")
	set(_report "${${_export_prefix}_REPORT_0}")
	set(_log "${${_export_prefix}_LOG_0}")
	set(_depfile "${${_export_prefix}_DEPFILE_0}")
	set(_cache_shader "${${_export_prefix}_CACHE_SHADER_0}")
	set(_cache_preprocess "${${_export_prefix}_CACHE_PREPROCESS_0}")
	set(_preprocessed "${${_export_prefix}_PREPROCESSED_0}")

	nbl_nsc_make_test_args(_cold_args "${_mode_base}_cold" "${_input}" "${_out}" "${_report}" "${_log}" "${_depfile}" "${_cache_shader}" "${_cache_preprocess}" "${_preprocessed}" "${_command}")
	nbl_nsc_make_test_args(_hit_args "${_mode_base}_hit" "${_input}" "${_out}" "${_report}" "${_log}" "${_depfile}" "${_cache_shader}" "${_cache_preprocess}" "${_preprocessed}" "${_command}")

	nbl_nsc_add_cache_tests(${_prefix} ${_fixture} "${_cold_args}" "${_hit_args}")
endfunction()

nbl_nsc_register_cache_layer(NBL_NSC_CACHE_SHADER nbl_nsc_shader_cache shader_cache "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_register_cache_layer(NBL_NSC_CACHE_PREPROCESS nbl_nsc_preprocess_cache preprocess_cache "${NBL_NSC_PREPROCESS_INPUT}" ${_EXPORT_PREPROCESS})
nbl_nsc_register_cache_layer(NBL_NSC_CACHE_PREAMBLE nbl_nsc_preamble_cache preamble_cache "${NBL_NSC_PREAMBLE_INPUT}" ${_EXPORT_PREAMBLE})

nbl_nsc_make_args_from_export(_NBL_NSC_SHADER_DISABLED_ARGS shader_cache_disabled "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_PREPROCESS_DISABLED_ARGS preprocess_cache_disabled "${NBL_NSC_PREPROCESS_INPUT}" ${_EXPORT_PREPROCESS})
nbl_nsc_make_args_from_export(_NBL_NSC_PREAMBLE_DISABLED_ARGS preamble_cache_disabled "${NBL_NSC_PREAMBLE_INPUT}" ${_EXPORT_PREAMBLE})
nbl_nsc_make_args_from_export(_NBL_NSC_ISOLATION_ARGS shader_cache_isolation "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_DEPS_ARGS deps_invalidation "${NBL_NSC_PREAMBLE_INPUT}" ${_EXPORT_PREAMBLE})
nbl_nsc_make_args_from_export(_NBL_NSC_PATH_ARGS path_normalization "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_RANDOM_DEFINES_ARGS random_defines "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_PARALLEL_ARGS parallel_smoke "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER} --parallel-jobs ${NBL_NSC_CACHE_TEST_PARALLEL_JOBS})
nbl_nsc_make_args_from_export(_NBL_NSC_STRESS_ARGS stress "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER} --iterations ${NBL_NSC_CACHE_TEST_ITERATIONS})
nbl_nsc_make_args_from_export(_NBL_NSC_REPORT_SCHEMA_ARGS report_schema "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_DEPFILE_ARGS depfile_contents "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_CACHE_OVERRIDE_ARGS cache_path_override "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_LARGE_GRAPH_ARGS large_include_graph "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_UNUSED_INCLUDE_ARGS unused_include "${NBL_NSC_SHADER_INPUT}" ${_EXPORT_SHADER})
nbl_nsc_make_args_from_export(_NBL_NSC_PREAMBLE_TIME_ARGS preamble_cache_hit_time "${NBL_NSC_PREAMBLE_INPUT}" ${_EXPORT_PREAMBLE} --budget-ms ${NBL_NSC_CACHE_PREAMBLE_BUDGET_MS})
nbl_nsc_make_no_cache_args(_NBL_NSC_NO_CACHE_ARGS "${NBL_NSC_PREAMBLE_INPUT}" ${_EXPORT_PREAMBLE})

nbl_nsc_add_single_test(NBL_NSC_CACHE_SHADER_DISABLED_TEST "${_NBL_NSC_SHADER_DISABLED_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_PREPROCESS_DISABLED_TEST "${_NBL_NSC_PREPROCESS_DISABLED_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_PREAMBLE_DISABLED_TEST "${_NBL_NSC_PREAMBLE_DISABLED_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_SHADER_ISOLATION_TEST "${_NBL_NSC_ISOLATION_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_DEPS_INVALIDATION_TEST "${_NBL_NSC_DEPS_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_PATH_NORMALIZATION_TEST "${_NBL_NSC_PATH_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_RANDOM_DEFINES_TEST "${_NBL_NSC_RANDOM_DEFINES_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_PARALLEL_SMOKE_TEST "${_NBL_NSC_PARALLEL_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_STRESS_TEST "${_NBL_NSC_STRESS_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_REPORT_SCHEMA_TEST "${_NBL_NSC_REPORT_SCHEMA_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_DEPFILE_CONTENTS_TEST "${_NBL_NSC_DEPFILE_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_PATH_OVERRIDE_TEST "${_NBL_NSC_CACHE_OVERRIDE_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_LARGE_GRAPH_TEST "${_NBL_NSC_LARGE_GRAPH_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_UNUSED_INCLUDE_TEST "${_NBL_NSC_UNUSED_INCLUDE_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_PREAMBLE_HIT_TIME_TEST "${_NBL_NSC_PREAMBLE_TIME_ARGS}")
nbl_nsc_add_single_test(NBL_NSC_CACHE_NO_CACHE_COLD_TEST "${_NBL_NSC_NO_CACHE_ARGS}")
set_tests_properties(NBL_NSC_CACHE_PREAMBLE_HIT_TIME_TEST PROPERTIES FIXTURES_REQUIRED nbl_nsc_preamble_cache)
