# NSC Toolchain Package

The general package model is described in [package-components.md](package-components.md).

This document focuses only on the `NSC` capability and its prebuilt-channel workflow.

## Exposed capability

The package namespace stays `Nabla::`.

The shader compiler toolchain capability is exposed as:

- component: `NSC`
- target: `Nabla::nsc`

Explicit consumption looks like:

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS NSC)
```

If the package root contains only the `NSC` capability, plain:

```cmake
find_package(Nabla CONFIG REQUIRED)
```

loads `NSC` automatically because the config package loads every capability present in the package root.

## Canonical layout used by the NSC bundle

The `NSC` bundle preserves the canonical Nabla package layout:

- `exe/tools/nsc/bin/nsc.exe`
- `runtime/nbl/Nabla.dll`
- `runtime/nbl/3rdparty/dxc/dxcompiler.dll`
- `cmake/NablaConfig.cmake`
- `cmake/NablaConfigVersion.cmake`
- `cmake/NablaNSCHelpers.cmake`
- `cmake/NablaNSCExportTargets*.cmake`

The `NSC` bundle is assembled from install tiles:

- `PackageConfig`
- `Runtimes`
- `NSCExecutables`
- `NSCConfig`

It does not use `RUNTIME_DEPENDENCY_SET` scanning and it does not invent a special package layout for the toolchain.

## Local build-system switch

General package-consumption details stay in [consume/README.md](consume/README.md).

This repository supports:

```cmake
-DNBL_NSC_MODE=PACKAGE
-DNBL_NSC_PACKAGE_ROOT=<optional-unpacked-package-root>
```

If `NBL_NSC_PACKAGE_ROOT` is set, the build system does:

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS NSC PATHS <root> NO_DEFAULT_PATH)
```

and uses `Nabla::nsc` only for shader compilation rules.

If `NBL_NSC_PACKAGE_ROOT` is empty, the build system uses NAM to materialize the package root into the build directory first and only then runs the same `find_package(...)` flow.

The current package-consumption path uses:

- manifests committed under `tools/nsc/manifests`
- a committed tag pin file named `<channel>.tag`
- a host-specific channel such as `nsc-windows-x64-release`
- flattened GitHub release assets published from `Devsh-Graphics-Programming/Nabla-Asset-Manifests`
- package reconstruction in the build directory through hardlinks when the host supports them

The flattened release asset naming is driven by NAM and uses:

- `<hex(relative-path)>__<basename>`

This lets the backend stay flat while the materialized result preserves the real Nabla package layout required by `find_package(Nabla COMPONENTS NSC)`.

By default the build system resolves the backend release tag from:

```text
tools/nsc/manifests/<channel>.tag
```

That file pins the immutable release tag currently selected by the branch.
If the file is missing, the build keeps the legacy fallback and uses the channel name itself as the release tag.

## Release channel notes

The release channel name may encode a host platform, for example:

- `nsc-windows-x64-release`

That channel name does not mean the package format itself is restricted to `Release` configuration only.

The package layout and install tiles still support normal multi-config staging.
The `release` suffix refers only to which published manifest channel is consumed by default for prebuilt toolchain distribution.

## Local packaging helper

For local validation this repository uses the maintainer helper shipped by NAM:

```text
cmake/nam/cmake/NablaAssetManifestsPrepareRelease.cmake
```

It takes an installed package root and emits:

- flattened payload files
- a `.dvc` manifest tree rooted at the selected channel
- one convenience manifests zip

Example:

```powershell
cmake `
  -D SOURCE_ROOT=build/dynamic/nsc-package `
  -D PAYLOAD_ROOT=build/dynamic/nsc-release/payload `
  -D MANIFEST_ROOT=tools/nsc/manifests `
  -D CHANNEL=nsc-windows-x64-release `
  -D MANIFESTS_ZIP=build/dynamic/nsc-release/nsc-windows-x64-release-manifests.zip `
  -D PRUNE=ON `
  -P cmake/nam/cmake/NablaAssetManifestsPrepareRelease.cmake
```

## CI packaging intent

CI may still build all matrix configurations.

The `NSC` channel staging step installs exactly these tiles into one prefix:

- `PackageConfig`
- `Runtimes`
- `NSCExecutables`
- `NSCConfig`

The intended release layout is:

- payload files published individually under a host-specific release tag such as `nsc-windows-x64-release`
- one convenience zip that contains only the generated `.dvc` manifest tree

The manifests zip is for maintainers.
Consumers read the unpacked `.dvc` files committed to this repository.

## Promotion flow

Repository-specific promotion details are described in [nsc-channel-promotion.md](nsc-channel-promotion.md).
