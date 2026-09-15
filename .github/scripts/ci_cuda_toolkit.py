#!/usr/bin/env python3

import argparse
import os
import platform
import subprocess
from pathlib import Path


REQUIRED_HEADERS = (
    "cuda.h",
    "nvrtc.h",
    "cuda_fp16.h",
    "vector_types.h",
)


def cuda_version() -> str:
    version = os.environ.get("CUDA_VERSION", "").strip()
    if not version:
        raise SystemExit("CUDA_VERSION is not set.")
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        raise SystemExit(f"CUDA_VERSION must start with major.minor, got: {version}")
    return version


def major_minor(version: str) -> str:
    major, minor, *_ = version.split(".")
    return f"{major}.{minor}"


def cache_base() -> Path:
    base = os.environ.get("CUDA_CACHE_BASE", "").strip()
    if not base:
        raise SystemExit("CUDA_CACHE_BASE is not set.")
    return Path(base)


def cache_root(version: str) -> str:
    return str(cache_base() / f"v{major_minor(version)}")


def cache_key(version: str) -> str:
    if platform.system() == "Windows":
        return f"cuda-toolkit-{version}-windows-2025-x64-v2"
    return f"cuda-toolkit-{version}-{platform.system().lower()}-x64-v1"


def cache_restore_key(version: str) -> str:
    if platform.system() == "Windows":
        return f"cuda-toolkit-{version}-windows-2025-x64-"
    return f"cuda-toolkit-{version}-{platform.system().lower()}-x64-"


def windows_install_root(version: str) -> Path:
    return Path(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{major_minor(version)}")


def emit_outputs() -> None:
    version = cuda_version()
    lines = (
        f"cache_root={cache_root(version)}",
        f"cache_key={cache_key(version)}",
        f"cache_restore_key={cache_restore_key(version)}",
    )
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as file:
            file.write("\n".join(lines))
            file.write("\n")
    else:
        print("\n".join(lines))


def nvcc_path(root: Path) -> Path:
    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    return root / "bin" / executable


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command))
    return subprocess.run(command, check=False, text=True, **kwargs)


def verify_toolkit(root: Path, version: str) -> bool:
    missing = [str(nvcc_path(root))]
    missing.extend(str(root / "include" / header) for header in REQUIRED_HEADERS)
    missing = [path for path in missing if not Path(path).exists()]
    if missing:
        print(f"CUDA Toolkit cache is incomplete at {root}.")
        for path in missing:
            print(f"missing: {path}")
        return False

    result = run([str(nvcc_path(root)), "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)
    expected = f"release {major_minor(version)}"
    if result.returncode != 0 or expected not in result.stdout:
        print(f"Expected CUDA Toolkit {major_minor(version)} at {root}.")
        return False
    return True


def verify() -> None:
    version = cuda_version()
    root = Path(os.environ.get("CUDA_TOOLKIT_ROOT", cache_root(version)))
    if not verify_toolkit(root, version):
        raise SystemExit(1)


def install() -> None:
    if platform.system() != "Windows":
        raise SystemExit("CUDA Toolkit install is only implemented for Windows CI.")

    version = cuda_version()
    install_root = windows_install_root(version)
    target_root = Path(cache_root(version))

    if verify_toolkit(target_root, version):
        print(f"CUDA Toolkit {major_minor(version)} already restored at {target_root}.")
        return

    result = run(["choco", "install", "cuda", "--version", version, "--yes", "--no-progress"])
    if result.returncode != 0:
        raise SystemExit(f"CUDA Toolkit {version} installation failed.")

    if not verify_toolkit(install_root, version):
        raise SystemExit(f"CUDA Toolkit {major_minor(version)} was not found at {install_root} after installation.")

    target_root.mkdir(parents=True, exist_ok=True)
    result = run(["robocopy", str(install_root), str(target_root), "/MIR", "/R:2", "/W:2", "/NFL", "/NDL", "/NP"])
    if result.returncode > 7:
        raise SystemExit(f"Failed to mirror CUDA Toolkit into cache root. robocopy exit code: {result.returncode}")

    if not verify_toolkit(target_root, version):
        raise SystemExit(f"CUDA Toolkit {major_minor(version)} was not found at {target_root} after installation.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("outputs", "install", "verify"))
    args = parser.parse_args()

    if args.command == "outputs":
        emit_outputs()
    elif args.command == "install":
        install()
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()
