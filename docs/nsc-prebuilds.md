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
- If you use `CAPS`, it generates a **cartesian product** of permutations and emits a `.spv` for each.
- A generated header (you choose the path via `INCLUDE`) containing:
  - a primary template `get_spirv_key<Key>(limits, features)` and `get_spirv_key<Key>(device)`
  - explicit specializations for each registered base `KEY`
  - the returned key already includes the build config prefix (compiled into the header).

Keys are strings that match the output layout:

```
<CONFIG>/<KEY>(.<capName>_<value>)(.<capName>_<value>)....spv
```

## The JSON "INPUTS" format

`INPUTS` is a JSON array of objects. Each object supports:

- `INPUT` (string, required): path to `.hlsl` (relative to `CMAKE_CURRENT_SOURCE_DIR` or absolute).
- `KEY` (string, required): base key (prefer without `.spv`; it is always appended, so using `foo.spv` will result in `foo.spv.spv`).
- `COMPILE_OPTIONS` (array of strings, optional): per-input extra options (e.g. `["-T","cs_6_8"]`).
- `DEPENDS` (array of strings, optional): per-input dependencies (extra files that should trigger rebuild).
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

Make sure shader inputs and includes are:

1. Marked as header-only on your target (so the IDE shows them, but the build system doesn't try to compile them with default HLSL rules like `fxc`):

```cmake
target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)
```

2. Listed as dependencies of the NSC custom commands (so editing any of them triggers a rebuild of the `.spv` outputs).

This is what the `DEPENDS` argument of `NBL_CREATE_NSC_COMPILE_RULES` (and/or per-input JSON `DEPENDS`) is for. Always include the main `INPUT` file itself and any files it includes; otherwise the build system might not re-run `nsc` when you change them.

## Minimal usage (no permutations)

Example pattern (as in `examples_tests/27_MPMCScheduler/CMakeLists.txt`):

```cmake
set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
  app_resources/common.hlsl
  app_resources/shader.comp.hlsl
)
target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

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
  DEPENDS ${DEPENDS}
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
auto key = nbl::this_example::builtin::build::get_spirv_key<"shader">(device);
auto bundle = assetMgr->getAsset(key.c_str(), loadParams);
```

`OUTPUT_VAR` (here: `KEYS`) is assigned the list of **all** produced access keys (all configurations + all permutations). This list is intended to be fed into `NBL_CREATE_RESOURCE_ARCHIVE(BUILTINS ${KEYS})`.

## Permutations via `CAPS`

`CAPS` lets you prebuild multiple SPIR-V variants parameterized by device limits or features.

Each `CAPS` entry looks like:

- `kind` (string, optional): `"limits"` or `"features"` (defaults to `"limits"` if omitted/invalid).
- `name` (string, required): identifier used in both generated HLSL config and C++ key (must be a valid C/C++ identifier).
- `type` (string, required): `bool`, `uint16_t`, `uint32_t`, `uint64_t`.
- `values` (array of numbers, required): the values you want to prebuild.
  - for `bool`, values must be `0` or `1`.

At build time, NSC compiles each combination of values (cartesian product). At runtime, `get_spirv_key` appends suffixes using the `limits`/`features` you pass in.

### Example: mixing `limits` and `features`

This example permutes over one device limit and one device feature (order matters: the suffix order matches the `CAPS` array order):

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

This produces `3 * 2 = 6` permutations per build configuration, and `KEYS` contains all of them (for example):

```
Debug/shader.maxComputeSharedMemorySize_16384.shaderFloat64_0.spv
Debug/shader.maxComputeSharedMemorySize_16384.shaderFloat64_1.spv
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

### CMake (`CMakeLists.txt`)

```cmake
include(common)

nbl_create_executable_project("" "" "" "")

set(OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/auto-gen")
set(DEPENDS
  app_resources/common.hlsl
  app_resources/shader.hlsl
)
target_sources(${EXECUTABLE_NAME} PRIVATE ${DEPENDS})
set_source_files_properties(${DEPENDS} PROPERTIES HEADER_FILE_ONLY ON)

set(JSON [=[
[
  {
    "INPUT": "app_resources/shader.hlsl",
    "KEY": "shader",
    "COMPILE_OPTIONS": ["-T", "lib_6_8"],
    "DEPENDS": [],
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
  DEPENDS ${DEPENDS}
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
