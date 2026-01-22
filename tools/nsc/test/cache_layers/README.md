# NSC Cache Layer Tests

This directory defines a cache-layer test suite for `nsc`.
The tests are driven by `tools/nsc/test/cache_layers/CMakeLists.txt` and the
Python runner `tools/nsc/test/cache_layers/cache_layers_test.py`.

## What is being tested

NSC has three cache layers that must stay correct and fast:

1) Shader cache
- Stores final SPIR-V.
- Hit path skips preprocess + compile and returns cached SPIR-V.

2) Preprocess cache
- Stores preprocessed source + dependency graph.
- Hit path skips full include processing, but still compiles.

3) Preamble cache (preamble/prefix cache)
- Stores preprocessed prefix for heavy include graphs.
- Hit path only preprocesses the body and reuses prefix.
- If the body has no preprocessor directives and no macro usage, it is passed through without running Wave.

## How the layers interact

The compile flow is:
- Shader cache probe (key + deps). On hit, return cached SPIR-V and skip all other work.
- If shader cache misses, probe preprocess cache.
  - If preprocess cache hits, compile using preprocessed code.
  - If preamble is enabled and available, only preprocess the body and reuse the cached prefix.
- On preprocess miss, do full preprocess + compile, then update caches.

All three layers together give:
- cold run: full preprocess + compile (safe baseline)
- body-only change: shader cache miss, preprocess cache hit, compile runs
- deps change: cold run (all caches miss)
- preamble hit: avoids re-lexing heavy includes on body edits

The build system enables all three by default, but each layer can be toggled
with CLI flags to verify behavior.

## Why use all three layers

- Shader cache gives the fastest hit path for unchanged inputs.
- Preprocess cache avoids re-walking includes when only the body changes.
- Preamble cache cuts Wave time for large include graphs even when the body changes.

Dropping any layer regresses a specific edit pattern. Using all three maximizes
iteration speed while keeping correctness, because every layer is validated by
its dependency tracking.

## Correctness and safety

Each cache entry is validated against its dependency graph and compilation
inputs. Any change in inputs, options, or includes invalidates the cache and
forces a cold run. When caches are enabled, `nsc` uses fast-safe validation:
mtime/size mismatches force a miss without hashing, so hits are never stale.
The tests do not enable "fast unsafe" paths.

## Test overview (CMake/CTest)

The suite defines a set of CTest entries with explicit cold/hit tests plus
additional integrity checks. All tests are executed in the same build
configuration you configured (`builtins` ON/OFF are respected automatically).

Core cache tests:
- `NBL_NSC_CACHE_SHADER_COLD_RUN_TEST`
- `NBL_NSC_CACHE_SHADER_HIT_TEST`
- `NBL_NSC_CACHE_PREPROCESS_COLD_RUN_TEST`
- `NBL_NSC_CACHE_PREPROCESS_HIT_TEST`
- `NBL_NSC_CACHE_PREAMBLE_COLD_RUN_TEST`
- `NBL_NSC_CACHE_PREAMBLE_HIT_TEST`

Extra correctness tests (no cross-config builds):
- Cache layer disable checks:
  - `NBL_NSC_CACHE_SHADER_DISABLED_TEST`
  - `NBL_NSC_CACHE_PREPROCESS_DISABLED_TEST`
  - `NBL_NSC_CACHE_PREAMBLE_DISABLED_TEST`
- Isolation / invalidation:
  - `NBL_NSC_CACHE_SHADER_ISOLATION_TEST`
  - `NBL_NSC_CACHE_DEPS_INVALIDATION_TEST`
- Path normalization:
  - `NBL_NSC_CACHE_PATH_NORMALIZATION_TEST`
- Randomized defines:
  - `NBL_NSC_CACHE_RANDOM_DEFINES_TEST`
- Parallel smoke (multi-process nsc calls with unique outputs):
  - `NBL_NSC_CACHE_PARALLEL_SMOKE_TEST`
- Stress (multiple repeated runs, timing stats only):
  - `NBL_NSC_CACHE_STRESS_TEST`
- Report schema sanity:
  - `NBL_NSC_CACHE_REPORT_SCHEMA_TEST`
- Depfile content check:
  - `NBL_NSC_CACHE_DEPFILE_CONTENTS_TEST`
- Cache override paths:
  - `NBL_NSC_CACHE_PATH_OVERRIDE_TEST`
- Large include graph:
  - `NBL_NSC_CACHE_LARGE_GRAPH_TEST`
- Unused include is excluded from depfile:
  - `NBL_NSC_CACHE_UNUSED_INCLUDE_TEST`
- Preamble hit timing (logs total time, optional budget):
  - `NBL_NSC_CACHE_PREAMBLE_HIT_TIME_TEST`
- No-cache cold baseline:
  - `NBL_NSC_CACHE_NO_CACHE_COLD_TEST`

## How it works

The tests compile a small HLSL shader that includes a local `proxy.hlsl`.
The proxy includes the same heavy builtins used in the cache test targets and
has injected markers for adding/removing `#define`s and include variants.

The Python runner:
- edits the body or proxy as required by a test mode
- runs `nsc` directly using the exact command line exported by
  `NBL_CREATE_NSC_COMPILE_RULES`
- reads the JSON report (`.spv.report.json`) to assert hit/miss and behavior

The JSON report fields used by the tests include:
- `shader_cache.hit`, `shader_cache.status`
- `preprocess_cache.status`, `preprocess_cache.hit`
- `preamble.enabled`, `preamble.used`
- `compile.called`

## Example timings (Release)

Measured from JSON reports (total_with_output_ms).
Cold-run and hit numbers are medians of 5 runs. Each hit sample is preceded by its cold-run seed.
Baseline is "No cache cold" per builtins mode. Relative vs no-cache is baseline / row.
Values below 1.0x mean slower than baseline.
Machine: AMD Ryzen 5 5600G with Radeon Graphics.
Config: Release, builtins OFF/ON (two baselines).
Includes stress: the proxy pulls three heavy builtins (intrinsics/matrix/vector). The full preprocessed output is ~11.3k lines (11274, measured from the Release preprocess-cache `.spv.pre.hlsl`).

Cold runs (no cache hits; preamble split can still be used):

| Scenario | Caches enabled | Preprocess path | total_with_output_ms (builtins OFF) | Relative vs no-cache (OFF) | total_with_output_ms (builtins ON) | Relative vs no-cache (ON) |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline no-cache cold | none | full preprocess | 1233 | 1.00x | 693 | 1.00x |
| Cold run (preprocess cache enabled) | shader + preprocess | full preprocess | 1449 | 0.85x | 912 | 0.76x |
| Cold run (all caches enabled) | shader + preprocess + preamble | full preprocess | 1276 | 0.97x | 748 | 0.93x |

Note: "Cold run (all caches enabled)" is still a cache miss and uses full preprocess; `preamble.used` stays false on miss. Small deltas between the cold rows (including an occasional slight speedup vs baseline) are measurement noise and OS caching effects.

Hit paths (caches enabled as configured by the test target):

| Scenario | Caches enabled | Hit path | total_with_output_ms (builtins OFF) | Speedup vs no-cache (OFF) | total_with_output_ms (builtins ON) | Speedup vs no-cache (ON) |
| --- | --- | --- | --- | --- | --- | --- |
| Shader cache hit | shader + preprocess + preamble | cached SPIR-V | 17 | 72.5x | 19 | 36.5x |
| Preprocess cache hit | shader + preprocess | preprocessed code + compile | 404 | 3.05x | 412 | 1.68x |
| Preamble cache hit | shader + preprocess + preamble | prefix reuse + body preprocess + compile | 219 | 5.63x | 251 | 2.76x |

These numbers are expected to vary across machines and drivers.
Builtins ON uses embedded archives, which reduces filesystem IO and typically improves cold-run times.

## Running the suite

From the build directory:

```
ctest -C Release -R NBL_NSC_CACHE_ --output-on-failure
```

Repeat runs for sampling:

```
ctest -C Release -R NBL_NSC_CACHE_ --repeat until-pass:5 --output-on-failure
```

## Tuning knobs

These are regular CMake cache variables:

- `NBL_NSC_CACHE_TEST_SEED`
  - Seed used for randomized define/body changes (0 = deterministic).
- `NBL_NSC_CACHE_TEST_ITERATIONS`
  - Number of iterations used by the stress test.
- `NBL_NSC_CACHE_TEST_PARALLEL_JOBS`
  - Number of parallel jobs used in the parallel smoke test.
- `NBL_NSC_CACHE_PREAMBLE_BUDGET_MS`
  - Optional budget for the preamble hit timing test (0 disables check).

## Build system defaults

The build system enables all three layers by default. CLI toggles exist for
validation:
- `-nbl-shader-cache`
- `-nbl-preprocess-cache`
- `-nbl-preprocess-preamble`

## Notes

- Tests are protected by a CTest `RESOURCE_LOCK` so they do not fight over the
  same inputs. The parallel smoke test uses unique outputs internally.
- The suite uses the current build configuration only; no extra Release/Debug
  builds are required.
