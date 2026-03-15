# NSC Channel Promotion

This document describes the Nabla-specific promotion flow for the prebuilt `NSC` toolchain channel.

General package composition stays in [package-components.md](package-components.md).
General package consumption stays in [consume/README.md](consume/README.md).
The generic release-preparation helper lives in NAM and is documented there.

## Source artifacts

The main build workflow produces two `NSC` artifacts for the host release channel:

- `<channel>-payload`
- `<channel>-manifests`

For the current Windows host channel the names are:

- `nsc-windows-x64-release-payload`
- `nsc-windows-x64-release-manifests`

The payload artifact contains the flattened per-file release assets.
The manifests artifact contains one convenience zip with the matching `.dvc` tree.

## Immutable release tags

Published channel releases live in `Devsh-Graphics-Programming/Nabla-Asset-Manifests`.

Each promoted release tag is immutable and uses the Nabla source commit that produced the payload:

```text
<channel>-<nabla_commit_sha>
```

Example:

```text
nsc-windows-x64-release-<nabla_commit_sha>
```

This keeps the backend pin stable across branches and does not depend on a GitHub Actions run id.

## Repo pin

The Nabla repo keeps two pinned inputs under `tools/nsc/manifests`:

- the committed `.dvc` tree for the selected channel
- a text file named `<channel>.tag` that stores the immutable release tag

For example:

```text
tools/nsc/manifests/nsc-windows-x64-release.tag
```

The build system reads that file and passes its contents to NAM as the backend release tag.

## Promote workflow

Promotion is manual only.
Nothing publishes automatically.

Use `.github/workflows/promote-nsc-channel.yml` with `workflow_dispatch`.

Inputs:

- `run_id`
  - workflow run that produced the source artifacts
- `target_branch`
  - Nabla branch that should receive the manifest update PR
- `channel`
  - manifest channel to promote

The workflow does four things:

1. downloads the `payload` and `manifests` artifacts from the selected Nabla run
2. publishes a release in `Nabla-Asset-Manifests` under `<channel>-<nabla_commit_sha>`
3. updates the committed `.dvc` tree and `<channel>.tag`
4. opens a PR against the requested `target_branch`

For production use the expected target is `master`.
For smoke tests the same workflow can target a branch such as `nsc-channel`.
