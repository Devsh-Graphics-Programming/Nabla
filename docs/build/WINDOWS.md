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

## CMake (DEPRICATED NOTES)

Best to use *cmake-gui*

Run *cmake-gui* and, as usually, give *CMake* root **Nabla** directory and where you want *CMake* to put project files and click "Configure" field. When *CMake* asks you to choose compiler/IDE, make sure to check whether there's a distinct option for *64bit* mode and, if this is the case, choose this one.

For single-config IDEs (Code::Blocks) you'll have to manually set `CMAKE_BUILD_TYPE` to `Debug` or `Release`. `Release` is default.

You also have options `BUILD_EXAMPLES` and `BUILD_TOOLS` which do exactly what they say. By "*tools*" you should currently understand just `convert2baw`.

For Windows *MSVC* required, *MinGW* build system maintenance will be delegated to the community.
