# Consuming Nabla Package

This document describes how to consume an installed Nabla package from another CMake project.

## 1. Package API

After `find_package(Nabla CONFIG REQUIRED)`, the package provides:

- imported target `Nabla::Nabla`
- helper `nabla_sync_runtime_modules(...)`
- helper `nabla_apply_runtime_lookup(...)`
- helper `nabla_setup_runtime_install_modules(...)`
- wrapper `nabla_setup_runtime_modules(...)`

On shared builds, runtime modules include Nabla and DXC.

Implementation and argument docs:

- package API implementation: `${Nabla_ROOT}/cmake/NablaConfig.cmake`
- source template in Nabla repo: `cmake/NablaConfig.cmake.in`
- each public helper has usage notes in comments directly above its definition

## 2. Minimal baseline

```cmake
cmake_minimum_required(VERSION 3.30)
project(MyApp CXX)

find_package(Nabla REQUIRED CONFIG)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Nabla::Nabla)
```

Behavior in this minimal setup:

- executable loads Nabla/DXC directly from package-provided lookup paths
- this works in consumer build interface without extra copy helpers
- install layout is not configured by this baseline

If you also need your own install layout, add install rules and relative lookup defines.
Helpers from sections below can do this for you.

## 3. Runtime setup primitives

### 3.1 Copy runtime modules

```cmake
nabla_sync_runtime_modules(
    TARGETS my_app
    MODE BUILD_TIME
    RUNTIME_MODULES_SUBDIR "Libraries"
)
```

or with explicit destination(s):

```cmake
nabla_sync_runtime_modules(
    DESTINATION_DEBUG "${CMAKE_BINARY_DIR}/Debug/Libraries"
    DESTINATION_RELEASE "${CMAKE_BINARY_DIR}/Release/Libraries"
    DESTINATION_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/Libraries"
    MODE CONFIGURE_TIME
)
```

Rules:

- use either `TARGETS` mode or `DESTINATION` / `DESTINATION_DEBUG` / `DESTINATION_RELEASE` / `DESTINATION_RELWITHDEBINFO` mode
- `MODE CONFIGURE_TIME` does copy during configure/generate
- `MODE BUILD_TIME` and `MODE BOTH` in destination mode require `BUILD_TRIGGER_TARGETS`

### 3.2 Apply runtime lookup defines

```cmake
nabla_apply_runtime_lookup(
    TARGETS my_app
    RUNTIME_MODULES_SUBDIR "Libraries"
)
```

This sets:

- `NBL_CPACK_PACKAGE_NABLA_DLL_DIR="./Libraries"`
- `NBL_CPACK_PACKAGE_DXC_DLL_DIR="./Libraries"`

### 3.3 Install runtime modules

```cmake
include(GNUInstallDirs)

nabla_setup_runtime_install_modules(
    RUNTIME_MODULES_SUBDIR "Libraries"
)

install(TARGETS my_app
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
```

## 4. Wrapper helper

`nabla_setup_runtime_modules(...)` composes:

- `nabla_sync_runtime_modules(...)`
- `nabla_apply_runtime_lookup(...)`
- optional `nabla_setup_runtime_install_modules(...)`

Example:

```cmake
nabla_setup_runtime_modules(
    TARGETS my_app
    MODE CONFIGURE_TIME
    RUNTIME_MODULES_SUBDIR "Libraries"
    INSTALL_RULES ON
)
```

## 5. Split flow global copy and per-exe lookup

This is the split pattern used by consumers that want one global copy setup and per-exe lookup:

```cmake
# one global copy setup
nabla_sync_runtime_modules(
    DESTINATION_DEBUG "${CMAKE_BINARY_DIR}/3rdparty/shared/Debug/Libraries"
    DESTINATION_RELEASE "${CMAKE_BINARY_DIR}/3rdparty/shared/Release/Libraries"
    DESTINATION_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/3rdparty/shared/RelWithDebInfo/Libraries"
    MODE CONFIGURE_TIME
)

# per executable target
nabla_apply_runtime_lookup(
    TARGETS my_app
    RUNTIME_MODULES_SUBDIR "Libraries"
)
```

## 6. Config mapping

Runtime source paths are resolved from mapped imported config of `Nabla::Nabla`.

Imported-config mapping applies automatically. This includes cross-config usage when one consumer config maps to a different imported config.

If you override mapping:

- do it in the same configure run
- if using `CMAKE_MAP_IMPORTED_CONFIG_<CONFIG>`, set it before `find_package(Nabla)`
- for `MODE CONFIGURE_TIME` and `MODE BOTH`, set mapping before helper call

## 7. Troubleshooting

### `Could not load dxcompiler module` or `Could not load Nabla API`

Check:

- lookup defines are applied to executable target(s)
- lookup subdir matches actual runtime layout
- runtime modules exist in build/install runtime directory

### Build works but installed app fails

Install rules are usually missing.

Use either:

- `nabla_setup_runtime_install_modules(...)`
- `nabla_setup_runtime_modules(... INSTALL_RULES ON)`

## 8. Design guidance

For relocatable consumers:

- keep lookup relative to executable
- never expose absolute paths in public compile definitions
- keep copy setup and lookup setup explicit in CMake

Note:

- current Nabla build interface still compiles some runtime lookup data with absolute paths
- this is a known issue on Nabla side and will be refactored
- do not propagate that pattern to package consumers
- consumer-facing package helpers are designed to avoid exposing absolute paths in consumer compile definitions

## 9. Smoke reference

`smoke/` is a reference consumer for Nabla package consumption.

It contains multiple usage flows:

- `MINIMALISTIC` link-only consumption without helper calls
- `CONFIGURE_ONLY` helper-based configure-time runtime sync
- `BUILD_ONLY` helper-based build-time runtime sync

Flow selection is done with `NBL_SMOKE_FLOW` in `smoke/CMakeLists.txt` and `FLOW` in `smoke/RunSmokeFlow.cmake`.

Smoke is also used as CI coverage for package consumption flows.  
The `smoke-tests` job in `.github/workflows/build-nabla.yml` runs those flows as end-to-end checks.
