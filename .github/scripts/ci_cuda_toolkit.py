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


def windows_paths(version: str) -> dict[str, str]:
    mm = major_minor(version)
    major, minor = mm.split(".")
    return {
        "cache_root": rf"C:\nabla-ci\cuda\v{mm}",
        "container_root": rf"C:\cuda\v{mm}",
        "container_root_cmake": f"C:/cuda/v{mm}",
        "version_env": f"CUDA_PATH_V{major}_{minor}",
        "cache_key": f"cuda-toolkit-{version}-windows-2025-x64-choco-v1",
    }


def windows_install_root(version: str) -> Path:
    return Path(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{major_minor(version)}")


def paths() -> dict[str, str]:
    version = cuda_version()
    if platform.system() == "Windows":
        return windows_paths(version)
    mm = major_minor(version)
    return {
        "cache_root": f"/opt/nabla-ci/cuda/v{mm}",
        "container_root": f"/cuda/v{mm}",
        "container_root_cmake": f"/cuda/v{mm}",
        "version_env": f"CUDA_PATH_V{mm.replace('.', '_')}",
        "cache_key": f"cuda-toolkit-{version}-{platform.system().lower()}-x64-v1",
    }


def emit_outputs() -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    values = paths()
    lines = [f"{key}={value}" for key, value in values.items()]
    if output:
        with open(output, "a", encoding="utf-8") as file:
            file.write("\n".join(lines))
            file.write("\n")
    else:
        print("\n".join(lines))


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(command))
    return subprocess.run(command, check=False, text=True, **kwargs)


def nvcc_path(root: Path) -> Path:
    executable = "nvcc.exe" if platform.system() == "Windows" else "nvcc"
    return root / "bin" / executable


def verify_toolkit(root: Path, version: str) -> bool:
    missing = [str(nvcc_path(root))]
    missing.extend(str(root / "include" / header) for header in REQUIRED_HEADERS)
    missing = [path for path in missing if not Path(path).exists()]
    if missing:
        print(f"CUDA Toolkit is incomplete at {root}.")
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
    values = paths()
    root = Path(os.environ.get("CUDA_TOOLKIT_ROOT", values["cache_root"]))
    if not verify_toolkit(root, version):
        raise SystemExit(1)


def install() -> None:
    if platform.system() != "Windows":
        raise SystemExit("CUDA Toolkit install is only implemented for Windows CI.")

    version = cuda_version()
    values = paths()
    install_root = windows_install_root(version)
    cache_root = Path(values["cache_root"])

    if verify_toolkit(cache_root, version):
        print(f"CUDA Toolkit {major_minor(version)} already restored at {cache_root}")
        return

    result = run(["choco", "install", "cuda", "--version", version, "--yes", "--no-progress"])
    if result.returncode != 0:
        raise SystemExit(f"CUDA Toolkit {version} installation failed.")

    if not verify_toolkit(install_root, version):
        raise SystemExit(f"CUDA Toolkit {major_minor(version)} was not found at {install_root} after installation.")

    cache_root.mkdir(parents=True, exist_ok=True)
    result = run(["robocopy", str(install_root), str(cache_root), "/MIR", "/R:2", "/W:2", "/NFL", "/NDL", "/NP"])
    if result.returncode > 7:
        raise SystemExit(f"Failed to mirror CUDA Toolkit into cache root. robocopy exit code: {result.returncode}")

    if not verify_toolkit(cache_root, version):
        raise SystemExit(f"CUDA Toolkit {major_minor(version)} was not found at {cache_root} after installation.")


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
