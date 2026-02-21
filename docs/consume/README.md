# Consuming Nabla Package

This document describes how to consume an installed Nabla package from another CMake project.

## 1. Package API

After `find_package(Nabla CONFIG REQUIRED)`, the package provides:

- imported target `Nabla::Nabla`
- helper `nabla_setup_runtime_modules(...)`
- helper `nabla_setup_runtime_install_modules(...)`

On shared builds, runtime modules include Nabla and DXC.

## 2. Locate the package

You can point CMake to the package with:

- `-D Nabla_DIR=<install-prefix>/cmake`
- `CMAKE_PREFIX_PATH=<install-prefix>`

Minimal baseline:

```cmake
cmake_minimum_required(VERSION 3.30)
project(MyApp CXX)

find_package(Nabla REQUIRED CONFIG)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Nabla::Nabla)
```

## 3. Flow NO_BUILD_COPY install to e.g. `./Libraries`

Use this flow when:

- build-time should load directly from package
- install tree should load from e.g. `./Libraries`

Call install-only helper:

```cmake
include(GNUInstallDirs)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Nabla::Nabla)

nabla_setup_runtime_install_modules(my_app
    RUNTIME_MODULES_SUBDIR "Libraries"
)

install(TARGETS my_app
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
```

What it does:

- adds runtime lookup defines `./Libraries`
- adds install rules for Nabla/DXC runtime modules to `${CMAKE_INSTALL_BINDIR}/Libraries`
- does not add post-build copy

Runtime behavior:

- build tree falls back to package runtime if `./Libraries` does not exist and relative package lookup can be resolved
- install tree uses `./Libraries` once modules are installed there

## 4. Flow WITH_BUILD_COPY install to e.g. `./Libraries`

Use one call when you want both:

- build-time copy to runtime subdir
- install-time copy to runtime subdir

```cmake
include(GNUInstallDirs)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Nabla::Nabla)

nabla_setup_runtime_modules(my_app
    RUNTIME_MODULES_SUBDIR "Libraries"
    INSTALL_RULES ON
)

install(TARGETS my_app
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
```

## 5. Config mapping

Runtime source paths are resolved from `$<TARGET_FILE:Nabla::Nabla>`.

Imported-config mapping applies automatically. This includes cross-config usage when one consumer config maps to a different imported config.

If you override mapping:

- do it in the same configure run
- if using `CMAKE_MAP_IMPORTED_CONFIG_<CONFIG>`, set it before `find_package(Nabla)`

## 6. Troubleshooting

### `Could not load dxcompiler module` or `Could not load Nabla API`

Check:

- helper usage matches your intended flow mode
- `RUNTIME_MODULES_SUBDIR` matches actual runtime folder layout
- install tree actually contains runtime modules under expected subdir

### Build works but installed app fails

Most often install rules are missing.

Use either:

- `nabla_setup_runtime_install_modules(...)` for `NO_BUILD_COPY`
- `nabla_setup_runtime_modules(... INSTALL_RULES ON)` for `WITH_BUILD_COPY`

### Build tree cannot resolve package runtime in install-only mode

This usually means your build tree and package runtime are on different roots or drives so a relative fallback cannot be formed.

Use one of:

- `nabla_setup_runtime_modules(... INSTALL_RULES ON)` to copy runtime modules into build tree

### Why modules are copied in build tree

Only `nabla_setup_runtime_modules(... INSTALL_RULES ON)` performs build-time copy.

If you want no build copy, use `nabla_setup_runtime_install_modules(...)` instead.

## 7. Design guidance

For relocatable consumers:

- keep lookup relative to executable
- never expose absolute paths in public compile definitions
- use one of the helper flows consistently per target
