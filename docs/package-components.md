# Nabla Package Components

This repository uses three separate concepts for package composition.

General consumer-facing package usage stays in [consume/README.md](consume/README.md).

## Package capabilities

Capabilities are what a consumer requests with:

```cmake
find_package(Nabla CONFIG REQUIRED [COMPONENTS ...])
```

The current capabilities are:

- `Core`
- `NSC`

`Core` is the base SDK capability.
It covers the normal Nabla package surface such as `Nabla::Nabla` and the shared package bootstrap needed to consume it.

If a consumer does not request explicit components, the package loads every capability that is physically present in the package root.

That means:

- an SDK-only package exposes `Core`
- an NSC-only package exposes `NSC`
- a combined package exposes both `Core` and `NSC`

## Install tiles

Install tiles are the physical subsets used to assemble package roots.

Each subsystem owns only its own tiles:

- top-level package bootstrap owns `PackageConfig`
- Nabla runtime and SDK own `Headers`, `Libraries`, `Runtimes`
- `tools/nsc` owns `NSCExecutables`, `NSCConfig`

This keeps package ownership local and avoids cross-directory install leaks.

The current shared package bootstrap tile is:

- `PackageConfig`
  - `cmake/NablaConfig.cmake`
  - `cmake/NablaConfigVersion.cmake`

The current NSC-specific tiles are:

- `NSCExecutables`
  - `exe/tools/nsc/bin/nsc.exe`
- `NSCConfig`
  - `cmake/NablaNSCExportTargets*.cmake`
  - `cmake/NablaNSCHelpers.cmake`

Shared tiles can be reused by multiple capabilities without changing the final package layout.
For example `NSC` reuses shared package bootstrap and runtime tiles but still owns only its own executable and config tiles.

## Bundle recipes

Bundle recipes are the staging-time unions of install tiles.

They are not the same thing as package capabilities.

For example:

$$ \text{NSC bundle} = \text{PackageConfig} \cup \text{Runtimes} \cup \text{NSCExecutables} \cup \text{NSCConfig} $$

and:

$$ \text{SDK bundle} = \text{PackageConfig} \cup \text{Headers} \cup \text{Libraries} \cup \text{Runtimes} $$

This preserves one canonical package layout while allowing different artifacts to reuse the same install tiles.

## CPack component dependencies

CPack component dependencies are still useful metadata for installers, but the CI packaging flow should treat bundle recipes explicitly.

In other words:

- capability selection is handled by `NablaConfig.cmake`
- install tiles are defined where files are owned
- bundle recipes are assembled by packaging logic
