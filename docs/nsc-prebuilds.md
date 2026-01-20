# NSC prebuilds (build-time HLSL -> SPIR-V)

This document explains how to use `NBL_CREATE_NSC_COMPILE_RULES` together with `NBL_CREATE_RESOURCE_ARCHIVE` to:

- Compile HLSL to SPIR-V at **build time** (via the `nsc` tool).
- Optionally generate **device-cap permutations** (limits/features "CAPS").
- Generate a small C++ header with **type-safe key getters** (`get_spirv_key<...>()`).
- Make the same code work with `NBL_EMBED_BUILTIN_RESOURCES` **ON** (embedded virtual archive) and **OFF** (mounted build directory) when loading your precompiled SPIR-V at runtime.

Definitions live in `cmake/common.cmake` (`NBL_CREATE_NSC_COMPILE_RULES`, `NBL_CREATE_RESOURCE_ARCHIVE`).

## Runtime mounting requirement (important)

All of this assumes your app mounts the directory/archive containing the NSC outputs (i.e. `BINARY_DIR`) into Nabla's virtual filesystem, then loads files via keys that are relative to that mounted root (the examples use `app_resources`).

The examples "just work" because they inherit from `nbl::examples::BuiltinResourcesApplication`, which mounts:

- `NBL_EMBED_BUILTIN_RESOURCES=OFF`: `system::CMountDirectoryArchive(NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT, ...)` at `app_resources`
- `NBL_EMBED_BUILTIN_RESOURCES=ON`: the generated embedded archive (e.g. `nbl::this_example::builtin::build::CArchive`) at `app_resources`

If you're writing your own app/extension and don't use `BuiltinResourcesApplication`, you must mount equivalently yourself (split by `NBL_EMBED_BUILTIN_RESOURCES`). Optionally set `IAssetLoader::SAssetLoadParams::workingDirectory` to whatever virtual root you want to load from.

The `MOUNT_POINT_DEFINE` argument of `NBL_CREATE_NSC_COMPILE_RULES` defines a C/C++ macro whose value is the absolute path to the NSC output directory (`BINARY_DIR`) that you mount when builtins are off (in examples it's `NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT`).

See `examples_tests/common/include/nbl/examples/common/BuiltinResourcesApplication.hpp` for the exact mounting logic.

## Why build-time NSC instead of runtime compilation?

Build-time compilation is usually preferable because it:

- Uses your build system's parallelism (Ninja/MSBuild jobs) to compile shaders quickly.
- Writes **only into the build tree** (no source tree pollution, easy clean/reconfigure).
- Lets CI validate "shaders compile" as part of a normal build.
- Enables fast runtime iteration: at runtime you only **pick** the right SPIR-V, you don't compile it.
- Makes shader compilation deterministic and reproducible (toolchain + flags captured by the build).

Runtime compilation is still useful for prototyping, but (assuming you don't use a runtime shader cache) it can make startup slower and shift failures to runtime instead of CI/build (a cache can hide the repeated cost on subsequent runs; our current one has some rough edges: it writes into the source tree and has issues when compiling many inputs from the same source directory).

## What `NBL_CREATE_NSC_COMPILE_RULES` produces

For each registered input it generates:

- One `.spv` output **per CMake configuration** (`Debug/`, `Release/`, `RelWithDebInfo/`).
- A matching `.spv.hash` sidecar for fast up-to-date checks on cache hits.
- If you use `CAPS`, it generates a **cartesian product** of permutations and emits a `.spv` for each.
- A generated header (you choose the path via `INCLUDE`) containing:
- a primary template `get_spirv_key<Key>(...args)` and `get_spirv_key<Key>(device, ...args)`
- `get_spirv_key` returns a small owning buffer; use `.view()` or implicit `std::string_view` to consume it
- arguments must follow the **kind order** as it appears in `CAPS` (first appearance), validated structurally by required member names/types for each kind (including `limits`/`features`, no strong typing)
  - `get_spirv_key<Key>(device, ...)` expects only **non-device** kinds in that same order; `limits`/`features` are injected from the device
  - note: an order-agnostic API would require enforcing unique member sets across kinds to guarantee unambiguous matching; we keep a conventional order instead to stay flexible without extra constraints
  - explicit specializations for each registered base `KEY`
  - the returned key already includes the build config prefix (compiled into the header).

Keys are hashed to keep filenames short and stable across long permutation strings. The **full key string** is built as:

```
<KEY>__<kind>.<capName>_<value>.<capName>_<value>...spv
```

Then `FNV-1a 64-bit` is computed from that full key (no `<CONFIG>` prefix), and the **final output key** is:

```
<CONFIG>/<hash>.spv
```

## The JSON "INPUTS" format

`INPUTS` is a JSON array of objects. Each object supports:

- `INPUT` (string, required): path to `.hlsl` (relative to `CMAKE_CURRENT_SOURCE_DIR` or absolute).
- `KEY` (string, required): base key (prefer without `.spv`; it is always appended, so using `foo.spv` will result in `foo.spv.spv`).
- `COMPILE_OPTIONS` (array of strings, optional): per-input extra options (e.g. `["-T","cs_6_8"]`).
- `DEPENDS` (array of strings, optional): extra per-input dependencies that are not discovered via `#include` (see below).
- `CAPS` (array, optional): permutation caps (see below).

You can register many rules in a single call, and you can call the function multiple times to append rules to the same `TARGET`.

## Compile options (generator expressions, defaults, debug info)

`NBL_CREATE_NSC_COMPILE_RULES` combines options from multiple sources:

- Built-in defaults from the helper (see `cmake/common.cmake`): HLSL version, Vulkan SPIR-V target env, scalar layout, warnings, and per-config optimization flags (e.g. `-O0` for Debug, `-O3` for Release) implemented via CMake generator expressions.
- Global extra options via `COMMON_OPTIONS` (CMake list).
- Per-input extra options via JSON `COMPILE_OPTIONS` (array of strings).

Both `COMMON_OPTIONS` and JSON `COMPILE_OPTIONS` support CMake generator expressions like `$<$<CONFIG:Debug>:...>` (the helper uses them itself), so you can make flags configuration-dependent when needed.

### Debug info for RenderDoc

The helper also exposes CMake options that append NSC debug flags **only for Debug config** (via generator expressions). Enable them if you want RenderDoc to show source/line information instead of just raw disassembly:

- `NSC_DEBUG_EDIF_FILE_BIT` (default `ON`) -> `-fspv-debug=file`
- `NSC_DEBUG_EDIF_TOOL_BIT` (default `ON`) -> `-fspv-debug=tool`
- `NSC_DEBUG_EDIF_SOURCE_BIT` (default `OFF`) -> `-fspv-debug=source`
- `NSC_DEBUG_EDIF_LINE_BIT` (default `OFF`) -> `-fspv-debug=line`
- `NSC_DEBUG_EDIF_NON_SEMANTIC_BIT` (default `OFF`) -> `-fspv-debug=vulkan-with-source`

## Source files and rebuild dependencies (important)

NSC supports depfiles and the CMake custom commands consume them, so **changes in any `#include`d HLSL file automatically trigger recompilation of the affected `.spv` outputs**. In most cases you no longer need to list includes manually.

Use `DEPENDS` only for **extra** inputs that are not discovered via `#include` (e.g. a generated header that is not included, a config file read by a custom include generator, or any non-HLSL file that should trigger a rebuild). You can register those extra dependencies if you need them, but in most projects `DEPENDS` should stay empty.

By default `NBL_CREATE_NSC_COMPILE_RULES` also collects `*.hlsl` files for IDE visibility. It recursively scans the current source directory (or `GLOB_DIR` if provided), adds those files as header-only, and groups them under `HLSL Files`. If you do not want this behavior, pass `DISCARD_DEFAULT_GLOB`.

- `GLOB_DIR` (optional): root directory for the default `*.hlsl` scan.
- `DISCARD_DEFAULT_GLOB` (flag): disables the default scan and IDE grouping.

## Cache layers (SPIR-V + preprocess)

There are three independent cache layers:

- `NSC_SHADER_CACHE` (default `ON`) -> SPIR-V cache (`<hash>.spv.ppcache`) for full compilation results.
- `NSC_SHADER_CACHE_COMPRESSION` (default `raw`) -> compression used for shader cache entries (`raw` or `lzma`).
- `NSC_PREPROCESS_CACHE` (default `ON`) -> preprocessor prefix cache (`<hash>.spv.ppcache.pre`) to avoid repeating Boost.Wave include work when only the main shader changes.
- `NSC_PREPROCESS_PREAMBLE` (default `ON`) -> preamble mode: reuse cached preprocessed prefix + macro state and run Wave only on the body, then compile without re-lexing the prefix.
- All layers are used only for compilation (not `-P` preprocess-only runs).
- When preprocess cache is enabled and used, NSC also writes a combined preprocessed view (`<hash>.spv.pre.hlsl`) next to the outputs.
  - This file is the exact input fed to DXC on the preprocess-cache path, so it's ready to paste into Godbolt for repros (use the same flags/includes).

With `-verbose`, `.log` shows:

- `Shader Cache: <path>` and `Cache hit!/miss! ...` for SPIR-V cache.
- `Preprocess cache: <path>` and `Preprocess cache hit!/miss! ...` for the prefix cache.
- Timing lines (performance):
  - `Shader cache load took: ...`
  - `Shader cache validate took: ...`
  - `Shader cache lookup took: ...`
  - `Shader cache write took: ...` (only when deps metadata changed on hit)
  - `Preprocess cache lookup took: ...`
  - `Total cache probe took: ...`
  - `Preamble body preprocess took: ...` (only when preamble mode is used)
  - `Preprocess took: ...` (only on compile path)
  - `Compile took: ...` (only on compile path)
  - `Total build time: ...` (preprocess + compile)
  - `Write output took: ...` (only when output file is written)
  - `Total took: ...` (overall tool runtime)

You can also toggle layers directly on the `nsc` CLI:

- `-nbl-shader-cache`
- `-nbl-shader-cache-compression <raw|lzma>`
- `-nbl-preprocess-cache`
- `-nbl-preprocess-preamble`
- `-nbl-stdout-log` (mirror the log file output to stdout)

Related CMake options:

- `NSC_PREPROCESS_PREAMBLE` (default `ON`)
- `NSC_STDOUT_LOG` (default `OFF`)
- `NSC_SHADER_CACHE_COMPRESSION` (default `raw`)

You can redirect the caches into a shared directory with:

- `NSC_CACHE_DIR` (path). The cache files keep the same relative layout as `BINARY_DIR` (including `<CONFIG>/<hash>`), but live under the given root. This is handy for CI or persistent cache volumes.

The preprocess cache key is based on the **prefix** of the input file (leading directives/comments plus forced includes), and cache validity is checked against include dependency hashes. That means:

- edits to the shader body still hit (fast path)
- changes to prefix directives, forced-includes, or included headers cause a cold run

## Minimal usage (no permutations)

Example pattern (as in `examples_tests/27_MPMCScheduler/CMakeLists.txt`):

```cmake
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(JSON [=[
[
  {
    "INPUT": "app_resources/shader.comp.hlsl",
    "KEY": "shader",
    "COMPILE_OPTIONS": ["-T", "cs_6_8"],
    "DEPENDS": [],
    "CAPS": []
  }
]
]=])

NBL_CREATE_NSC_COMPILE_RULES(
  TARGET ${EXECUTABLE_NAME}SPIRV
  LINK_TO ${EXECUTABLE_NAME}
  BINARY_DIR ${OUTPUT_DIRECTORY}
  MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
  COMMON_OPTIONS -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VAR KEYS
  INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
  NAMESPACE nbl::this_example::builtin::build
  INPUTS ${JSON}
)
```

Then include the generated header and use the key to load the SPIR-V:

```cpp
#include "nbl/this_example/builtin/build/spirv/keys.hpp"
// ...
auto keyBuf = nbl::this_example::builtin::build::get_spirv_key<"shader">(device);
std::string_view key = keyBuf;
auto bundle = assetMgr->getAsset(key.data(), loadParams);
```

`OUTPUT_VAR` (here: `KEYS`) is assigned the list of **all** produced access keys (all configurations + all permutations). These are already hashed (e.g. `Debug/123456789.spv`) and are intended to be fed into `NBL_CREATE_RESOURCE_ARCHIVE(BUILTINS ${KEYS})`.

## Permutations via `CAPS`

`CAPS` lets you prebuild multiple SPIR-V variants parameterized by device limits or features.

Each `CAPS` entry looks like:

- `kind` (string, optional): `"limits"`, `"features"`, or `"custom"` (defaults to `"limits"` if omitted/invalid).
- `struct` (string, required for `kind="custom"`): name of the custom permutation struct (valid C/C++ identifier). If you use `limits` or `features` here, do not also use the built-in `limits`/`features` kinds in the same rule.
- `name` (string, required): identifier used in both generated HLSL config and C++ key (must be a valid C/C++ identifier).
- `type` (string, required): `bool`, `uint16_t`, `uint32_t`, `uint64_t`, `int16_t`, `int32_t`, `int64_t`, `float`, `double`.
- `values` (array of numbers, required): the values you want to prebuild.
  - for `bool`, values must be `0` or `1`.
  - for signed integer types, negative values are allowed.
  - for `float`/`double`, you can provide **numbers or numeric strings** (e.g. `-1`, `-1.0`, `1e-3`, or `-1.f` for floats). Values are **normalized** to canonical scientific notation (1 digit before the decimal, 8 digits after for `float` or 16 for `double`, signed exponent with 2 or 3 digits). The normalized text becomes part of the key.

At build time, NSC compiles each combination of values (cartesian product). At runtime, `get_spirv_key` appends suffixes using the structs you pass in for `limits`/`features` (duck-typed by required members) and any custom kinds. Each group starts with `__limits`, `__features`, or `__<customStruct>`, followed by `.member_<value>` entries. Group order follows the **first appearance of each kind in `CAPS`** (and this same order is the required argument order for `get_spirv_key`); groups with no members are omitted.

Each generated `.config` file defines a `DeviceConfigCaps` struct for HLSL. It includes:
- flat members for `limits`/`features` (backwards compatibility with older shaders)
- nested structs for custom kinds only, e.g. `DeviceConfigCaps::userA`

Example shape:

```hlsl
struct DeviceConfigCaps
{
    NBL_CONSTEXPR_STATIC_INLINE uint32_t maxImageDimension2D = 16384u;
    NBL_CONSTEXPR_STATIC_INLINE bool shaderCullDistance = true;

    struct userA
    {
        NBL_CONSTEXPR_STATIC_INLINE uint32_t mode = 0u;
        NBL_CONSTEXPR_STATIC_INLINE uint32_t quality = 1u;
    };
};
```

For more complex usage and regression-style checks (constexpr vs runtime, hashing, mixed payloads), see `examples_tests/73_SpirvKeysTest`.

### Grouping caps by kind (optional)

To avoid repeating the same `kind`, you can group caps with `members`:

```cmake
set(JSON [=[
[
  {
    "INPUT": "app_resources/shader.hlsl",
    "KEY": "shader",
    "COMPILE_OPTIONS": ["-T", "lib_6_8"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "userA",
        "members": [
          { "name": "mode", "type": "uint32_t", "values": [0, 1] },
          { "name": "quality", "type": "uint32_t", "values": [1, 2, 4] }
        ]
      },
      {
        "kind": "features",
        "members": [
          { "name": "shaderFloat64", "type": "bool", "values": [0, 1] }
        ]
      }
    ]
  }
]
]=])
```

### Example: mixing `limits` and `features`

This example permutes over one device limit and one device feature. Suffix order follows the `CAPS` order (`__limits` then `__features` here), and member order within each group follows the `CAPS` order for that group:

```cmake
set(JSON [=[
[
  {
    "INPUT": "app_resources/shader.hlsl",
    "KEY": "shader",
    "COMPILE_OPTIONS": ["-T", "lib_6_8"],
    "DEPENDS": ["app_resources/common.hlsl"],
    "CAPS": [
      {
        "kind": "limits",
        "name": "maxComputeSharedMemorySize",
        "type": "uint32_t",
        "values": [16384, 32768, 65536]
      },
      {
        "kind": "features",
        "name": "shaderFloat64",
        "type": "bool",
        "values": [0, 1]
      }
    ]
  }
]
]=])

NBL_CREATE_NSC_COMPILE_RULES(
  # ...
  OUTPUT_VAR KEYS
  INPUTS ${JSON}
)
```

## Custom permutation structs

If you need permutations based on data outside of device `limits`/`features`, define a custom struct in C++ and use `kind: "custom"` with `struct` set to the parameter name. At runtime you can pass any struct type that exposes the required members with matching types; **argument order follows the `CAPS` kind order**. Using custom names `limits` or `features` is allowed, but you cannot mix them with the built-in `limits`/`features` kinds in the same rule.

Example:

```cmake
set(JSON [=[
[
  {
    "INPUT": "app_resources/fft.hlsl",
    "KEY": "fft",
    "COMPILE_OPTIONS": ["-T", "cs_6_8"],
    "CAPS": [
      {
        "kind": "custom",
        "struct": "fftConfig",
        "name": "passCount",
        "type": "uint32_t",
        "values": [4, 8]
      }
    ]
  }
]
]=])

NBL_CREATE_NSC_COMPILE_RULES(
  # ...
  OUTPUT_VAR KEYS
  INPUTS ${JSON}
)
```

Runtime usage:

```cpp
nbl::this_example::FFTConfig cfg = {};
cfg.passCount = 4;
auto key = nbl::this_example::builtin::build::get_spirv_key<"fft">(device, cfg);
```

Constexpr usage with extra structs (order must match `CAPS` kind order, first appearance):

```cpp
struct MyLimits { uint32_t maxImageDimension2D; };
struct MyFeatures { bool shaderCullDistance; };
struct UserA { uint32_t mode; uint32_t quality; };
struct UserB { bool useAlternatePath; bool useFastPath; };

constexpr UserA userA = { 0u, 1u };
constexpr UserB userB = { false, true };
constexpr MyLimits limits = { 16384u };
constexpr MyFeatures features = { true };

static constexpr auto keyBuf =
    nbl::this_example::builtin::build::get_spirv_key<"shader_cd">(userA, userB, limits, features);
static constexpr std::string_view keyView = keyBuf;

```

## Common pitfalls

- Argument order must follow the **first appearance of each kind in `CAPS`**; this is an intentional convention to keep the API flexible.
- `get_spirv_key` returns a buffer; prefer `std::string_view key = buf;` or `buf.view()` to consume it.
- Do not store a `std::string_view` from a temporary buffer; keep the buffer alive.
- `float`/`double` CAP values are normalized to canonical scientific notation (1 digit before the decimal, 8 or 16 digits after, signed exponent); values passed to `get_spirv_key` must match one of the CAP values exactly.
- `constexpr` key generation works with `float`/`double` members when the values match the CAP list.

This produces `3 * 2 = 6` permutations per build configuration, and `KEYS` contains all of them (for example):

```
Debug/6014683721143225910.spv
Debug/10493750182651038558.spv
...
```

Practical tip: for numeric limits you often want to "bucket" real device values into one of the prebuilt values. The CountingSort example does exactly that:

- CMake definition: `examples_tests/10_CountingSort/CMakeLists.txt`
- Runtime bucketing: `examples_tests/10_CountingSort/main.cpp`

```cpp
auto limits = m_physicalDevice->getLimits();
constexpr std::array<uint32_t, 3u> AllowedMaxComputeSharedMemorySizes = { 16384, 32768, 65536 };

auto upperBoundSharedMemSize = std::upper_bound(
	AllowedMaxComputeSharedMemorySizes.begin(), AllowedMaxComputeSharedMemorySizes.end(), limits.maxComputeSharedMemorySize
);
// devices which support less than 16KB of max compute shared memory size are not supported
if (upperBoundSharedMemSize == AllowedMaxComputeSharedMemorySizes.begin())
{
	m_logger->log("maxComputeSharedMemorySize is too low (%u)", ILogger::E_LOG_LEVEL::ELL_ERROR, limits.maxComputeSharedMemorySize);
	exit(0);
}

limits.maxComputeSharedMemorySize = *(upperBoundSharedMemSize - 1);

auto key = nbl::this_example::builtin::build::get_spirv_key<"prefix_sum_shader">(limits, m_physicalDevice->getFeatures());
```

## Pairing with `NBL_CREATE_RESOURCE_ARCHIVE` (works with builtins ON/OFF)

The recommended pattern is to always call `NBL_CREATE_RESOURCE_ARCHIVE` right after the NSC rules, using the produced `KEYS` list:

```cmake
NBL_CREATE_RESOURCE_ARCHIVE(
  TARGET ${EXECUTABLE_NAME}_builtinsBuild
  LINK_TO ${EXECUTABLE_NAME}
  BIND ${OUTPUT_DIRECTORY}
  BUILTINS ${KEYS}
  NAMESPACE nbl::this_example::builtin::build
)
```

### How `BINARY_DIR`, `MOUNT_POINT_DEFINE`, and `BIND` fit together

- In `NBL_CREATE_NSC_COMPILE_RULES`, `BINARY_DIR` is the output directory where NSC writes the compiled files:
  - `${BINARY_DIR}/<CONFIG>/<KEY>....spv`
- In `NBL_CREATE_NSC_COMPILE_RULES`, `MOUNT_POINT_DEFINE` is the *name* of a C/C++ preprocessor define whose value is set to the **absolute path** of `BINARY_DIR`.
  - Example: `MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT` results in something like `-DNBL_THIS_EXAMPLE_BUILD_MOUNT_POINT="C:/.../auto-gen"` on the target.
  - Keys returned by `get_spirv_key<...>()` are relative to that directory; the full path on disk is:
    - `${NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT}/<key>`
- In `NBL_CREATE_RESOURCE_ARCHIVE`, `BIND` should point at the same directory as `BINARY_DIR`.
  - The `BUILTINS` list entries must be relative to `BIND`.
  - This is why pairing it with `OUTPUT_VAR KEYS` works: `KEYS` is exactly the list of relative paths under `BINARY_DIR` that were generated by the NSC rules, so the archive generator knows what to serialize/embed.

This is designed to work in both modes:

- `NBL_EMBED_BUILTIN_RESOURCES=OFF`:
  - `NBL_CREATE_RESOURCE_ARCHIVE` becomes a no-op (creates a dummy interface target).
  - You load SPIR-V from the **build directory** mounted into the virtual filesystem.
  - `MOUNT_POINT_DEFINE` provides an absolute path (e.g. `NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT`) for mounting.
- `NBL_EMBED_BUILTIN_RESOURCES=ON`:
  - `NBL_CREATE_RESOURCE_ARCHIVE` generates a small library that embeds the listed files into a virtual archive and emits `.../CArchive.h` under the requested `NAMESPACE`.
  - You mount the embedded archive instead of a directory; runtime loading code stays the same (keys don't change).

## Notes / gotchas

- `INCLUDE` must be a **relative** path (it is emitted under the build tree and added to include dirs automatically).
- Prefer not to include `.spv` in `KEY` (the extension is appended unconditionally); if you do, you'll just get `.spv.spv` in the final filename/key (not an error, just not what you want).
- You can mix:
  - per-input `COMPILE_OPTIONS` (inside JSON), and
  - global `COMMON_OPTIONS` (CMake list after `COMMON_OPTIONS`).

## Troubleshooting (no logs / silent NSC failures)

Sometimes an NSC compile rule fails during the build, but the build output doesn't show a useful log. In that case, run the failing command under a debugger:

1. Open the generated Visual Studio solution and set the `nsc` project/target as the Startup Project.
2. Open the `nsc` project properties and set **Debugging -> Command Arguments**.
3. Copy the exact CLI from the failing "NSC Rules" custom command (the one that calls `nsc.exe`) into the Command Arguments field.
4. Start debugging (`F5`) and reproduce; if needed, put a breakpoint in the HLSL compiler/preprocessor codepath and step until you find the root cause.

If the error looks like a preprocessing issue, note that we use Boost.Wave as the preprocessor; it can have quirky edge cases (e.g. needing a trailing newline/whitespace at the end of a file for correct parsing).

## Best practices

- Prefer compiling to a shader library (`-T lib_6_x`) and using multiple entry points when possible: fewer inputs means fewer compile rules and less build overhead; at runtime you still choose the entry point from the same `.spv`.
- Treat `CAPS` as a build-time cost multiplier (cartesian product). If the permutation count gets too large (thousands+), prebuilding usually stops paying off; an example of such workload is `examples_tests/23_Arithmetic2UnitTest`.

## Complete example (expand)

<details>
<summary>NSC rules + archive + runtime key usage</summary>

NSC emits depfiles and the custom commands consume them, so changes in `#include`d HLSL files automatically trigger recompilation of the affected outputs. In most cases you do not need to list includes manually. Use `DEPENDS` only for extra inputs that are not discovered via `#include`.

### CMake (`CMakeLists.txt`)

```cmake
include(common)

nbl_create_executable_project("" "" "" "")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")

set(JSON [=[
[
  {
    "INPUT": "app_resources/shader.hlsl",
    "KEY": "shader",
    "COMPILE_OPTIONS": ["-T", "lib_6_8"],
    "CAPS": [
      {
        "kind": "limits",
        "name": "maxComputeSharedMemorySize",
        "type": "uint32_t",
        "values": [16384, 32768, 65536]
      },
      {
        "kind": "features",
        "name": "shaderFloat64",
        "type": "bool",
        "values": [0, 1]
      }
    ]
  }
]
]=])

NBL_CREATE_NSC_COMPILE_RULES(
  TARGET ${EXECUTABLE_NAME}SPIRV
  LINK_TO ${EXECUTABLE_NAME}
  BINARY_DIR ${OUTPUT_DIRECTORY}
  MOUNT_POINT_DEFINE NBL_THIS_EXAMPLE_BUILD_MOUNT_POINT
  COMMON_OPTIONS -I ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VAR KEYS
  INCLUDE nbl/this_example/builtin/build/spirv/keys.hpp
  NAMESPACE nbl::this_example::builtin::build
  INPUTS ${JSON}
)

# Works for both NBL_EMBED_BUILTIN_RESOURCES=ON/OFF
NBL_CREATE_RESOURCE_ARCHIVE(
  NAMESPACE nbl::this_example::builtin::build
  TARGET ${EXECUTABLE_NAME}_builtinsBuild
  LINK_TO ${EXECUTABLE_NAME}
  BIND ${OUTPUT_DIRECTORY}
  BUILTINS ${KEYS}
)
```

### Runtime usage (C++)

```cpp
#include "nbl/this_example/builtin/build/spirv/keys.hpp"

// Load relative to the VFS mount (examples mount it at "app_resources")
asset::IAssetLoader::SAssetLoadParams lp = {};
lp.workingDirectory = "app_resources";

auto limits = device->getPhysicalDevice()->getLimits();
limits.maxComputeSharedMemorySize = 32768; // one of the prebuilt values; real code should bucket/clamp with std::upper_bound (see the CountingSort snippet above)

auto key = nbl::this_example::builtin::build::get_spirv_key<"shader">(limits, device->getEnabledFeatures());
auto bundle = assetMgr->getAsset(key.c_str(), lp);
const auto assets = bundle.getContents();
auto spvShader = asset::IAsset::castDown<asset::IShader>(assets[0]);

// params.shader.shader = spvShader.get();

// If you compiled with `-T lib_6_x`, pick the entry point at pipeline creation time (e.g. `params.shader.entryPoint = "main";`).
```

</details>
