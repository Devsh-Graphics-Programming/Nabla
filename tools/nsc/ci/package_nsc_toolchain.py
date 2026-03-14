#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True)
    parser.add_argument("--payload-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--manifests-zip", required=True)
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def flat_release_asset_name(relative_path: str) -> str:
    encoded = relative_path.encode("utf-8").hex()
    basename = relative_path.split("/")[-1]
    return f"{encoded}__{basename}"


def iter_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dvc_manifest(path: Path, file_name: str, md5: str, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        "outs:\n"
        f"- md5: {md5}\n"
        f"  size: {size}\n"
        "  hash: md5\n"
        f"  path: {file_name}\n"
    )
    path.write_text(contents, encoding="utf-8", newline="\n")


def zip_tree(source_root: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in iter_files(source_root):
            archive.write(file_path, file_path.relative_to(source_root).as_posix())


def main() -> int:
    args = parse_args()

    package_root = Path(args.package_root).resolve()
    payload_root = Path(args.payload_root).resolve()
    manifest_root = Path(args.manifest_root).resolve()
    manifests_zip = Path(args.manifests_zip).resolve()
    channel_root = manifest_root / args.channel

    if not package_root.is_dir():
        raise SystemExit(f"package root does not exist: {package_root}")

    if args.clean:
        shutil.rmtree(payload_root, ignore_errors=True)
        shutil.rmtree(channel_root, ignore_errors=True)
        if manifests_zip.exists():
            manifests_zip.unlink()
        if args.metadata_json:
            metadata_path = Path(args.metadata_json).resolve()
            if metadata_path.exists():
                metadata_path.unlink()

    payload_root.mkdir(parents=True, exist_ok=True)
    channel_root.mkdir(parents=True, exist_ok=True)

    metadata: list[dict[str, object]] = []
    seen_assets: dict[str, str] = {}

    for file_path in iter_files(package_root):
        relative_path = file_path.relative_to(package_root).as_posix()
        asset_name = flat_release_asset_name(relative_path)
        if asset_name in seen_assets:
            raise SystemExit(
                f"flat release asset collision: {asset_name} for {relative_path} and {seen_assets[asset_name]}"
            )
        seen_assets[asset_name] = relative_path

        destination = payload_root / asset_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)

        md5 = compute_md5(file_path)
        size = file_path.stat().st_size
        manifest_path = channel_root / f"{relative_path}.dvc"
        write_dvc_manifest(manifest_path, file_path.name, md5, size)

        metadata.append(
            {
                "relative_path": relative_path,
                "release_asset": asset_name,
                "md5": md5,
                "size": size,
            }
        )

    zip_tree(manifest_root, manifests_zip)

    if args.metadata_json:
        metadata_path = Path(args.metadata_json).resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8", newline="\n")

    print(f"Packaged {len(metadata)} files from {package_root}")
    print(f"Payload root: {payload_root}")
    print(f"Manifest root: {channel_root}")
    print(f"Manifest zip: {manifests_zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
