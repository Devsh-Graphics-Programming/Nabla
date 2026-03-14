# NSC Toolchain Package

This repository can expose the shader compiler toolchain as a dedicated package component without importing the rest of the Nabla SDK into a consumer build graph.

## Package shape

The package namespace stays `Nabla::`.

The shader compiler toolchain is exposed as:

- component: `NSC`
- target: `Nabla::nsc`

The regular SDK remains the default result of:

```cmake
find_package(Nabla CONFIG REQUIRED)
```

The toolchain-only path is:

```cmake
find_package(Nabla CONFIG REQUIRED COMPONENTS NSC)
```

This loads only `Nabla::nsc`.

The `NSC` component packages a fixed runtime layout for the tool itself:

- `exe/nsc.exe`
- `exe/Nabla*.dll`
- `exe/dxcompiler.dll`
- `cmake/NablaConfig.cmake`
- `cmake/NablaNSCExportTargets*.cmake`

The package does not rely on `RUNTIME_DEPENDENCY_SET` scanning.
It installs the known Nabla and DXC runtime DLLs explicitly.

## Release-only consumption

The `NSC` package can contain only the `Release` variant of `nsc`.

When that happens, the config package maps consumer configs to `Release` automatically:

- `Debug -> Release`
- `RelWithDebInfo -> Release`
- `MinSizeRel -> Release`

This lets a Debug or RelWithDebInfo application use a fast release `nsc` binary without building a debug compiler toolchain locally.

## Local build-system switch

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

It does not import the Nabla SDK targets into the active Nabla build tree.

If `NBL_NSC_PACKAGE_ROOT` is empty, the build system uses NAM to materialize
the package root into the build directory first and only then runs the same
`find_package(...)` flow.

The current package-consumption path uses:

- manifests committed under `tools/nsc/manifests`
- a host-specific channel such as `nsc-windows-x64-release`
- flattened GitHub release assets
- package reconstruction in the build directory through hardlinks when the host
  supports them

The flattened release asset naming is driven by NAM and uses:

- `<hex(relative-path)>__<basename>`

This lets the backend stay flat while the materialized result preserves the
real Nabla package layout required by `find_package(Nabla COMPONENTS NSC)`.

## Local packaging helper

For local validation this repository also carries:

```text
tools/nsc/ci/package_nsc_toolchain.py
```

It takes an installed `NSC` component and emits:

- flattened payload files
- a `.dvc` manifest tree rooted at the selected channel
- one convenience manifests zip

Example:

```powershell
python tools/nsc/ci/package_nsc_toolchain.py `
  --package-root build/dynamic/nsc-package `
  --payload-root build/dynamic/nsc-release/payload `
  --manifest-root tools/nsc/manifests `
  --channel nsc-windows-x64-release `
  --manifests-zip build/dynamic/nsc-release/nsc-windows-x64-release-manifests.zip `
  --clean
```

## CI packaging intent

CI can still build all matrix configurations.

The developer-facing channel should publish only the `Release` `NSC`
toolchain bundle.

The intended release layout is:

- payload files published individually under a host-specific release tag such
  as `nsc-windows-x64-release`
- one convenience zip that contains only the generated `.dvc` manifest tree

The manifest zip is for maintainers only.
Consumers read the unpacked `.dvc` files committed to this repository.
