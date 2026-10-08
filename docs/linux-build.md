# Building Nabla on Linux

This guide details the prerequisites, toolchain setup, configuration, compilation, and testing procedures for Nabla on Linux (tested on Ubuntu 24.04 LTS).

---

## 1. Prerequisites & Toolchain Setup

### System Packages
Install the required build tools, XCB/X11 development libraries, and utilities via your package manager:

```bash
sudo apt update
sudo apt install -y ninja-build lld nasm python3 git git-lfs ccache \
  libxcb-xkb-dev libxcb-randr0-dev libxcb-xinput-dev libxcb-icccm4-dev \
  libxcb-keysyms1-dev libxcb-cursor-dev libxcb-xfixes0-dev libxkbcommon-x11-dev \
  vulkan-tools
git lfs install
```

### Compiler: Clang 20
Nabla on Linux requires **Clang 20** with LLD (Clang 18 encounters compiler crashes on `CAssetConverter.cpp`). GCC is not currently supported (no `CXX_GNU` profile).

- **Option A (APT package)**: If `clang-20` is installed via APT:
  ```bash
  export CC=clang-20
  export CXX=clang++-20
  ```
- **Option B (Standalone LLVM archive)**: If using a standalone LLVM/Clang release:
  ```bash
  export CC=/path/to/llvm-20/bin/clang
  export CXX=/path/to/llvm-20/bin/clang++
  ```

### CMake (≥ 3.31)
The root `CMakeLists.txt` requires CMake version ≥ 3.31. Because Ubuntu 24.04 LTS defaults to CMake 3.28, install CMake 3.31+ or 4.x (tested with CMake 4.3.5) from the Kitware binary distribution or APT repository. Ensure the newer `cmake` executable is available in your `PATH`.

### Vulkan SDK (1.4.x)
Nabla targets Vulkan 1.3+ / 1.4. While the engine vendors `Vulkan-Headers` and loads Vulkan dynamically via Volk, runtime execution and validation require a Vulkan 1.4 driver and the LunarG Vulkan SDK (tested with LunarG SDK **1.4.363.0**).

Download and extract the SDK tarball, then activate the environment in your shell:
```bash
source /path/to/VulkanSDK/1.4.x.x/setup-env.sh
```

---

## 2. Git Submodules Initialization

Nabla vendors dependencies (DXC, Boost, OpenEXR, glslang, shaderc, Vulkan-Headers, etc.) and example suites via git submodules.

### Fresh Fork Submodule Hydration

`.gitmodules` points every submodule at an absolute URL. Upstream uses relative URLs for `3rdparty/boost/superproject` and `docker/msvc-winsdk`, and in a fork those resolve under the fork owner (`raydelto`), where they don't exist; `linux-port` uses the absolute `Devsh-Graphics-Programming` URLs instead (NAB-10). `Ditt-Reference-Scenes` is a private reference repository that must be excluded.

Exclude the private scenes, pick the protocol (SSH or HTTPS), and initialize:

```bash
# In your clone / worktree of raydelto/Nabla:
git checkout linux-port

# Clones made before NAB-10 still carry the old relative URLs in .git/config; refresh them once:
git submodule sync -- 3rdparty/boost/superproject docker/msvc-winsdk

# Exclude private scenes and initialize recursively (use HTTPS rewrite if SSH keys are not set up):
git -c fetch.parallel=0 \
    -c url.https://github.com/.insteadOf=git@github.com: \
    -c submodule."Ditt-Reference-Scenes".update=none \
    submodule update --init --recursive
```

### Local Submodule Caching (Fast Path)

When working across multiple local worktrees or checkouts, use the cached initialization helper to clone directly from an existing populated clone on disk without network downloads (~15 seconds):

```bash
cmake/scripts/linux/init-submodules-cached.sh <path-to-populated-cache> .
```

> **Note on examples fork pin**: Ensure the local cache clone has fetched the `raydelto` remote in its `examples_tests` submodule before hydrating:
> ```bash
> git -C <path-to-populated-cache>/examples_tests fetch raydelto
> ```
> This ensures that the integration pin `c337709e` from `raydelto/Nabla-Examples-and-Tests` is present in the cache's git object database.

---

## 3. CMake Configuration

Configure Nabla with the **Ninja** generator (required by the DirectX Shader Compiler sub-build on Linux) and Clang:

```bash
cmake -S . -B build/linux-clang-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_LINKER_TYPE=LLD \
  -DNBL_NSC_MODE=SOURCE \
  -DNBL_BUILD_EXAMPLES=ON \
  -DNBL_PCH=ON \
  -DNBL_ENABLE_DOCKER_INTEGRATION=OFF \
  -DNBL_UPDATE_GIT_SUBMODULE=OFF \
  -D_NBL_JOBS_AMOUNT_=2
```

### Key Configuration Flags:
- `-G Ninja`: **Mandatory**. The nested DXC sub-build enforces Ninja on Linux.
- `-DCMAKE_LINKER_TYPE=LLD`: Links using LLVM's `lld` linker.
- `-DNBL_NSC_MODE=SOURCE`: Builds the Nabla Shader Compiler (`nsc`) CLI from source.
- `-DNBL_BUILD_EXAMPLES=ON`: Enables building the examples and unit tests suite.
- `-DNBL_PCH=ON`: Enables precompiled headers.
- `-DNBL_ENABLE_DOCKER_INTEGRATION=OFF`: Disables Docker container packaging.
- `-DNBL_UPDATE_GIT_SUBMODULE=OFF`: Prevents CMake from re-updating pre-hydrated submodules.
- `-D_NBL_JOBS_AMOUNT_=2`: Throttles nested build job concurrency.

---

## 4. Build Execution & Memory Throttling

> [!WARNING]
> **RAM Constraint Warning**: Compiling Microsoft DXC (LLVM/Clang) and Boost templates consumes extensive memory. Always limit DXC compilation concurrency to 2 parallel threads (`-j2`). Unbounded parallel builds can exhaust system RAM and cause out-of-memory errors.

### Step 1: Build DirectX Shader Compiler (`dxcompiler`)
```bash
cmake --build build/linux-clang-release -j2 --target dxcompiler
```
Produces `build/linux-clang-release/3rdparty/dxc/build/lib/libdxcompiler.so`.

### Step 2: Build Nabla Engine and NSC
```bash
cmake --build build/linux-clang-release -j4 --target nsc
```
Produces `build/linux-clang-release/src/nbl/libNabla.so` and `tools/nsc/bin/nsc`.

### Step 3: Run NSC Test Suite
Verify `nsc` compiler functionality and runtime discovery:
```bash
(cd build/linux-clang-release/tools/nsc && ctest --output-on-failure)
```
Expect: `100% tests passed, 0 tests failed out of 6`.

### Step 4: Build Examples and Unit Tests
```bash
cmake --build build/linux-clang-release -j4 --target \
  01_hellocoresystemasset \
  02_hellocompute \
  21_lrucacheunittest \
  23_arithmetic2unittest
```
Executables are placed into `examples_tests/<TestName>/bin/`.

---

## 5. Running and Validating Tests

Before running GPU executables, make sure the Vulkan SDK environment is activated:
```bash
source /path/to/VulkanSDK/1.4.x.x/setup-env.sh
```

Optionally enable validation layers or select a specific GPU ICD:
```bash
# Enable Khronos validation layer and loader logging
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LOADER_DEBUG=layer,driver

# Select GPU vendor ICD (examples):
# export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json   # NVIDIA
# export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.json    # Intel Mesa
```

Execute each binary using a subshell `(cd ... && ./)` so the working directory of your shell is preserved:

### 1. `01_HelloCoreSystemAsset` (VFS & Async Assets)
```bash
(cd examples_tests/01_HelloCoreSystemAsset/bin && ./01_hellocoresystemasset)
```
- Tests VFS mounting, archive extraction, async I/O futures, and image encoding/decoding.
- Expected result: **Exit code 0** (clean pass on both NVIDIA and Intel setups).

### 2. `02_HelloCompute` (Vulkan 1.4 Compute & BDA)
```bash
(cd examples_tests/02_HelloCompute/bin && ./02_hellocompute)
```
- Compiles HLSL compute kernel at runtime via `libdxcompiler.so`, dispatches 524,288 threads using Buffer Device Addresses, synchronizes with timeline semaphores, and validates memory readback.
- Expected result: **Exit code 0** on both NVIDIA RTX and Intel UHD GPUs.

### 3. `21_LRUCacheUnitTest` (Core Data Structures)
```bash
(cd examples_tests/21_LRUCacheUnitTest/bin && ./21_lrucacheunittest)
```
- Stress tests `nbl::core::ResizableLRUCache` allocations and eviction callbacks.
- Expected result: **Exit code 0**.

### 4. `23_Arithmetic2UnitTest` (GPU Workgroup & Subgroup Parallel Math)
```bash
(cd examples_tests/23_Arithmetic2UnitTest/bin && ./23_arithmetic2unittest)
```
- Cross-validates GPU parallel reductions, inclusive scans, and exclusive scans against CPU ground truth across various subgroup and workgroup sizes.
- **NVIDIA GPU**: **Exit code 0** (~7 minutes runtime).
- **Intel UHD (Mesa ANV)**: Known driver bug — exits with **code 139** (segfault inside `libvulkan_intel.so` during `vkCreateComputePipelines` on native subgroup size 32 inclusive scan at workgroup size 64). Emulated subgroup sizes and native subgroup sizes 8/16 all pass. An upstream bug report to Mesa is pending (Troubleshooting Issue 21).
