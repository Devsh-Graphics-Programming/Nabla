# Windows build

## Supported toolsets

- **[MSVC](https://visualstudio.microsoft.com/pl/downloads/)**

## Build modes

### Vanilla

Most extensions disabled.

### CUDA

#### **Additional required dependencies:**

- **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**

**Nabla** only supports *CUDA* interop using the Driver API not the Runtime API. We use NVRTC to produce runtime compiled PTX.

Because *CUDA* needs its own version the GeForce (Quadro or Tesla) Driver, its often a few minor versions behind your automatically updated Windows driver, the install will fail even if it prompts you to agree to installing an older driver. 

So basically first remove your driver, then install the *CUDA SDK*. You can also thank NVidia for making the CUDA SDK a whole whopping 2.5 GB on Windows.

### Optix

#### **Additional required dependencies:**

- **CUDA build mode's dependencies**

- **[OptiX SDK](https://developer.nvidia.com/designworks/optix/download)** 

After dealing with installing *CUDA* install *Optix SKD*.

## CMake

### Visual Studio 17 2022 generator (DEPRICATED NOTES)

Best to use *cmake-gui*

Run *cmake-gui* and, as usually, give *CMake* root **Nabla** directory and where you want *CMake* to put project files and click "Configure" field. When *CMake* asks you to choose compiler/IDE, make sure to check whether there's a distinct option for *64bit* mode and, if this is the case, choose this one.

For single-config IDEs (Code::Blocks) you'll have to manually set `CMAKE_BUILD_TYPE` to `Debug` or `Release`. `Release` is default.

You also have options `BUILD_EXAMPLES` and `BUILD_TOOLS` which do exactly what they say. By "*tools*" you should currently understand just `convert2baw`.

For Windows *MSVC* required, *MinGW* build system maintenance will be delegated to the community.

### Ninja Multi-Config generator

The target toolset is MSVC. Before invoking CMake you need to [initialize VS's environment](https://learn.microsoft.com/en-us/cpp/build/cmake-presets-vs?view=msvc-170#sourcing-the-environment-when-building-with-command-line-generators-on-windows) by calling

```cmd
vcvarsall.bat <platform>
```

in your cmd (eg. `vcvarsall.bat x64` call will initialize environment for amd64, for more info go [here](https://learn.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170#vcvarsall-syntax)). Once initialization is done you can use Ninja CMake preset to build Nabla, eg.

```cmd
cmake --build --preset user-build-dynamic-ninja-multi --config Debug -- --quiet
```

The build preset will pick configure preset associated with it and invoke configure then build. Pay attention that if you don't initialize VS's environment the following will fail to compile with MSVC toolset.
