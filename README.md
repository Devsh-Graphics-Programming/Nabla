# Nabla

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Nabla** (previously called **[IrrlichtBaW](https://github.com/buildaworldnet/IrrlichtBAW)** ) is a new renovated version of older **[Irrlicht](http://irrlicht.sourceforge.net/)** engine. 
The name change to Nabla allows for using Nabla side by side with the legacy Irrlicht and IrrlichtBaW engines. 
The project currently aims for a thread-able and *Vulkan*-centered API, the Vulkan backend is almost complete, and OpenGL and ES backends are currently in maintenance mode. 

This framework has been kindly begun by the founder ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** of **[Devsh Graphics Programming Sp. z O.O.](http://devsh.eu/)**  and was almost entirely sponsored by **Build A World Aps** in its early days, now it has been picked up by **[Ditt B.V.](https://www.ditt.nl/)**.

## (Get Hired) Jobs and Internships

If you are a programmer with a passion for High Performance Computing, Mathematics and Computer Graphics

If you can be in charge of your own time managment and work 4-day work weeks 100% remotely

Then make something impressive using **Nabla**, open a PR and contact us (`jobs@devsh.eu` or **[discord](https://discord.gg/4MTCVaN)**) with your CV.

We would also be happy to sponsor your master or bachelor thesis as long as:
- You are an above average student with an interest in Graphics
- It will be written in English
- It will produce contributions to Nabla which we can license under Apache 2.0

For internships contact us with:
- CV in english
- A neat description of any paperwork you'll need (schools/universities)
_Most importantly contact us at least 3 weeks in advance of your internship organisation deadline!_

## (Hire Us) Contracting

The members of **Devsh Graphics Programming Sp. z O.O.** (Company Registration (KRS) #: 0000764661) are available (individually or collectively) for contracts on projects of various scopes and timescales, especially on foreign frameworks, codebases and third-party 3D frameworks. 

**We provide expertise in:**

 - OpenGL
 - OpenGL ES 
 - WebGL
 - WebGPU
 - Vulkan 
 - OpenCL 
 - CUDA 
 - D3D12 and D3D11 
 - computer vision
 - Audio programming
 - DSP
 - video encoding and decoding
 - High Performance Computing

Our language of choice is C++20 however we're also amenable to C#, Java, Python and related languages.

Contact `newclients@devsh.eu` with inquires into contracting.

## Showcase

### Screenshots

### <u>Our Production Mitsuba Compatible Path Tracer made for Ditt B.V.</u>

Currently working on the `ditt` branch, in the process of being ported to Vulkan KHR Raytracing.

You can download a stable build [here](https://artifactory.devsh.eu/Ditt/ci/data/artifacts/public/Ditt.tar.bz2)

![](https://github.com/Devsh-Graphics-Programming/Nabla-Site-Media/blob/master/media/readme/screenshots/4e86cf2b-3f8e-40eb-9835-6553ea205df2.jpg?raw=true)

### [Multiple Importance Sampling and Depth of Field](https://www.youtube.com/watch?v=BuyVlQPV7Ks)

![](https://github.com/Devsh-Graphics-Programming/Nabla-Site-Media/blob/master/media/readme/gifs/myballs/Multiple%20Importance%20Sampling%20and%20Depth%20of%20Field%203.gif?raw=true)

![](https://github.com/Devsh-Graphics-Programming/Nabla-Site-Media/blob/master/media/readme/gifs/myballs/Multiple%20Importance%20Sampling%20and%20Depth%20of%20Field%205.gif?raw=true)

## Main Features

- **Frontend API with Vulkan as First Class Citizen**
- **Thread safe and context pollution safe OpenGL**
- **Asset management pipeline**
- **Automatic pipeline layout creation**
- **Shader introspection**
- **Using SPIR-V shaders in OpenGL and ES**
- **Libraries of GLSL shader functions**
- **Compute shaders**
- **Virtual Texturing**
- **Virtual Geometry (programmable and non programmble fetching) with triangle batching**
- **CUDA and Vulkan interop**
- **CPU asset manipulation (image filtering, image format transcoding, mesh optimization and manipulation)**
- **GPU driven Scene Graph**
- **Material Compiler for Path Tracing UberShaders**

## Main Delivered Extensions

- **Auto Exposure**
- **Tonemapper**
- **Mitsuba scene loader (auto-generated shaders)** 
- **Fastest blur on the planet** 
- **OptiX interop**
- **Bullet physics beginner integration**
- **GPU Radix Sort**

## Platforms

- [x] **Windows**

- [x] **Linux**

- [x] **Android 7.0 +**

- [ ] **Mac OS**

- [ ] **iOS**

## Build summary

|    ![][BUILD_STATUS]     |   Release    |     RWDI     |    Debug     |
| :----------------------: | :----------: | :----------: | :----------: |
|   **Windows MSVC x64**   | ![][MSVC_1]  | ![][MSVC_2]  | ![][MSVC_3]  |
| **Android Clang x86_64** | ![][CLANG_1] | ![][CLANG_2] | ![][CLANG_3] |
|    **Linux GCC x64**     |   ![][NA]    |   ![][NA]    |   ![][NA]    |

[MSVC_1]: https://ci.devsh.eu/buildStatus/icon?job=BuildNabla%2FDEVSH_CONFIGURATIONS%3DRelease%2CDEVSH_NODES%3Dpredator15%2CDEVSH_OS%3DWindows
[MSVC_2]: https://ci.devsh.eu/buildStatus/icon?job=BuildNabla%2FDEVSH_CONFIGURATIONS%3DRelWithDebInfo%2CDEVSH_NODES%3Dpredator15%2CDEVSH_OS%3DWindows
[MSVC_3]: https://ci.devsh.eu/buildStatus/icon?job=BuildNabla%2FDEVSH_CONFIGURATIONS%3DDebug%2CDEVSH_NODES%3Dpredator15%2CDEVSH_OS%3DWindows
[CLANG_1]: https://ci.devsh.eu/buildStatus/icon?job=BuildNabla%2FDEVSH_CONFIGURATIONS%3DRelease%2CDEVSH_NODES%3Dnode1%2CDEVSH_OS%3DAndroid
[CLANG_2]: https://ci.devsh.eu/buildStatus/icon?job=BuildNabla%2FDEVSH_CONFIGURATIONS%3DRelWithDebInfo%2CDEVSH_NODES%3Dnode1%2CDEVSH_OS%3DAndroid
[CLANG_3]: https://ci.devsh.eu/buildStatus/icon?job=BuildNabla%2FDEVSH_CONFIGURATIONS%3DDebug%2CDEVSH_NODES%3Dnode1%2CDEVSH_OS%3DAndroid
[NA]: https://img.shields.io/badge/free%20slot-n%2Fa-red
[BUILD_STATUS]: https://img.shields.io/badge/build-status-blueviolet

# Quick start

### Use Nabla from package

TODO - soon

### Build Nabla with Docker

TODO - soon

### Build Nabla manually

#### Minimal required dependencies

- **[CMake 3.29.2](https://cmake.org/download/)** version or higher
- **[Vulkan SDK 1.2.198.1](https://vulkan.lunarg.com/sdk/home)** version or higher
- **[NASM 2.15](https://www.nasm.us/pub/nasm/releasebuilds/?C=M;O=D)** version or higher
- **[Python 3.10.2](https://www.python.org/downloads/release/python-3102/)** version or higher

### Cloning the project

```shell
git clone git@github.com:Devsh-Graphics-Programming/Nabla.git <target directory>
```

#### Force HTTPS protocol (optional)

We support cloning Nabla with **ssh only**, however you can still force clone with https for whole repository and it's all submodules by overriding project git config setup.

```powershell
git init
git config --project protocol.*.allow always
git config --project url."https://github.com/".insteadOf "git@github.com:"
git remote add origin https://github.com/Devsh-Graphics-Programming/Nabla.git
git fetch origin master
git checkout master
```

### Configure & Generate with CMake

#### Target platform

Check individual instructions for building for a particular platform in **./docs/build** directory before going further. For each platform it is assumed minimal required dependencies are installed and additional may be specified depending on the target.

#### Submodules

CMake will update **all required submodules** for you by default but it doesn't mean the initialization & update attempt will be successful (it is often not when performed on a shaky internet connection, so you may end up with dirty, locked or un-versioned submodules) - if not successful then try to re-configure CMake again. We have a few options for managing submodules by CMake, for more details and description check the [update submodule script](https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/cmake/submodules/update.cmake). For example sometimes it may appear that there won't be any files in submodules' directories because of an update fail or dirty git cache, you could fix it and bring them back by enabling `NBL_FORCE_ON_UPDATE_GIT_SUBMODULE` CMake option. Note that we do not recommend to initialize and update submodules by hand in cmd because of private submodules in the repository.

#### More options

You can disable generating projects for examples with `NBL_BUILD_EXAMPLES`. It's recommended to build Nabla as shared library however you can also build as static library by turning on `NBL_STATIC_BUILD`.

#### CMake presets

We have customised [presets](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html) which may be handy to configure **Nabla** project for a target platform with predefined & common configurations.

# Caveats and Particular Behaviour

## Hardcoded Caps

### Max Descriptor Sets is always 4

## Debugging with RenderDoc

###  Non-programmatic OpenGL catpures will be delimited inconsistently

Due to our no-pollution opengl state isolation policy, we have 1 queue or swapchain = 1 thread = 1 gl context + 1 master context and thread for device calls.

Renderdoc therefore serializes all calls, and presents them inside the capture in interleaved order (records them on a single timeline "as they happened").

Furthermore it has no idea what constitutes a frame, because swap-buffers call happens on a separate thread than all the other API calls. **So use the `IGPUQueue` start/end capture methods!**

### RenderDoc flips images for display in the ImageViewer tab on OpenGL captures

Ctrl+F `localRenderer in https://github.com/baldurk/renderdoc/blob/4103f6a5455b9734e9bf74e254577f5c03188136/renderdoc/core/image_viewer.cpp

### OpenGL/Vulkan Inconsistencies

In certain cases same calls to Vulkan and OpenGL might result in y-flipped image relevant to the other API.

Both APIs write (-1,-1) in NDC space to (0,0) in image space (two wrongs make right), and memory-wise (0,0) always represents the lowest byte in memory.

This inconsistency comes from swapchain presentation. When presenting the swapchain, the image location (0,0) corresponds to **bottom-left** in OpenGL and **top-left** in Vulkan.

#### Solution by Surface Transforms

We solve this inconsistency by using [surface transforms](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkSurfaceTransformFlagBitsKHR.html); This transforms are relative to `presentation engine’s natural orientation`. and we report `HORIZONTAL_MIRROR_180` support in our OpenGL backend and defer handling these rotations (relative to natural orientaion) to the user.

We provide helper functions in both GLSL and C++ Nabla codebase to consider surface transforms, See [surface_transform.glsl](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/builtin/glsl/utils/surface_transform.glsl)

Note that it is common to apply surface transformation to projection matrices to account for this fact. See [getSurfaceTransformationMatrix](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/video/surface/ISurface.h) and [Android Developers Guide to Pre-rotation](https://developer.android.com/games/optimize/vulkan-prerotation)

Use [`ISwapchain getSurfaceTransform()`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/video/ISwapchain.h) to get the transformation from swapchain.

- When generating projection matricies, take into account the aspect ratio (which is changed when rotating 90 or 270 degrees). For this, we have helper functions in both GLSL and the ISurface class:
    - [`float getTransformedAspectRatio(const E_SURFACE_TRANSFORM_FLAGS transform, uint32_t w, uint32_t h)`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/video/surface/ISurface.h)
    - [`nbl_glsl_surface_transform_transformedExtents`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/builtin/glsl/utils/surface_transform.glsl)

- On the swapchain rendering pass, perform **one** of the following transforms:
    - If rendering **directly to the swapchain**, you can apply the (post) transform matrix to your projection or combined view-projection matrix **for rendering** (don't pre-multiply with projection matrix for use outside rendering):
        - [`matrix4SIMD ISurface::getSurfaceTransformationMatrix(const E_SURFACE_TRANSFORM_FLAGS transform)`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/video/surface/ISurface.h)
        - [`nbl_glsl_surface_transform_applyToNDC`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/builtin/glsl/utils/surface_transform.glsl) (This takes in an NDC coordinate and multiplies it with the transform matrix in one function)

    - If using `imageStore` to write **directly to the swapchain**, you can either:
        - Apply a transform to the screen-space coordinates being written to the swapchain:
            - [`nbl_glsl_surface_transform_applyToScreenSpaceCoordinate`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/builtin/glsl/utils/surface_transform.glsl)
        - Apply an **inverse** transform to the screen-space coordinates (taken from `gl_GlobalInvocationID.xy`) before turning them into UV/world-space coordinates for rendering:
            - [`nbl_glsl_surface_transform_applyInverseToScreenSpaceCoordinate`](https://github.com/Devsh-Graphics-Programming/Nabla/tree/master/include/nbl/builtin/glsl/utils/surface_transform.glsl)

# Continuous integration

## Nabla Python Framework

### CPU & GPU local tests shipped with the repository

#### Description

**Nabla** aims to implement full CPU & GPU tests used by our CI in-house infrastructure groovy pipelines in relocatable way allowing users to execute and debug the tests locally on their own devices as well. Each test is a Python module which part of it gets created with the CMake build system configuration. A test is defined and created as a module by

- top json configuration file
- interface Python script

Top json configuration file contains run & build info, array of profiles and json inputs for a Python testing module. The file is processed and validated by CMake to create output profiles bound to the Python module. Single profile contains data used by the module to execute tests with and the implementation of the test is located in interface Python script - the script overrides common abstract interface defined as a integral part of Nabla Python framework module. The description is abstract and doesn't contain specific details, join our discord if you have any questions!

#### Development & Debug

Each valid Python module contains autogenerated .vscode's *launch.json* and *settings.json* to make development easy. You just need to `Open with Code`in the module's directory. Make sure to install [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) there to enable IntelliSense (Pylance), Linting, Debugging (multi-threaded, remote), code formatting, refactoring, unit tests, and more.

A target for which Python module is available for testing lists a special Python Framework section located in its solution's project file structure (**Visual Studio only!**) containing Nabla Python Framework sources, target's interface Python script, autogenerated json profiles and autogenerated` __main__.py` module script. Unfortunately pure Visual Studio works so-so with Python (issues with search module paths defined as `searchPath` .pyproj's property, issues with IntelliSense and Debugging) - we encourage to use Visual Studio Code if you need to Debug an interface or develop it.

#### Runtime

##### Command line

To run all tests bound to a module with a command line you just need to execute `python3 -m <module_reference>`. You can also specify special arguments to have more control over the test execution, for more details see [template module script](https://github.com/Devsh-Graphics-Programming/Nabla-Continous-Integration-Python-Framework/blob/4f7a67a3fa9bb418bcb07fa2f7a5853e55b853c4/scripts/__main__.py.cmake)

##### Visual Studio

You need to open a target's solution. To launch all tests bound to a module a module's `__main__.py` file located in Python Framework section must be selected and `Debug -> Execute File in Python interactive` executed.

**Visual Studio Code**

You need to open module's directory as workspace with `Open with Code `, select `Run and Debug` icon, select `__main__.py` file and run it.

# License

**Nabla** is released under the **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)** license. See [**LICENSE.md**](https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/LICENSE.md) for more details.

# Documentation

**(WIP/TODO)**

# Official Support (Discord)

Permanent members of *Devsh Graphics Programming Sp. z O.O.* use this to organise publicly visible work. **[Join to the server](https://discord.gg/4MTCVaN)** to get into more details.

# Credits and Attribution

#### The authors of **Nabla** are:

- **Mateusz Kielan** ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** (Lead Programmer)
- **Arkadiusz Lachowicz** ***[@AnastaZIuk](https://github.com/AnastaZIuk)*** (Associate Graphics Programmer/Senior Build System Engineer)
- **Erfan Ahmadi [@Erfan](https://github.com/Erfan-Ahmadi)** (Mid Programmer)
- **Achal Pandey [@achalpandeyy](https://github.com/achalpandeyy)** (Associate Programmer)
- **Przemysław Pachytel** ***[@Przemog1](https://github.com/Przemog1)*** (Junior Programmer/Technical Writer)

#### Past Authors and Contributors:

- **Krzysztof Szenk** ***[@crisspl](https://github.com/Crisspl)*** (Senior Programmer: Everything in Nabla has been touched by his golden hands!)
- **Danylo Sadivnychyi [@sadiuk](https://github.com/sadiuk)** (Junior Programmer: Android system interfaces and buildsystem, FFT Ocean)
- **Cyprian Skrzypczak** ***[@Hazardu](https://github.com/Hazardu)*** (embeded resources and optimalizations)
- ***[@khom-khun](https://github.com/khom-khun)*** (Bullet Physics Extension + Example and **[the irrBaW-test repository of easy to understand demos](https://github.com/khom-khun/irrBAW-test)**)
- **Søren Gronbech** 
- ***[@florastamine](https://github.com/florastamine)*** **Nguyễn Ngọc Huy** (sRGB-Correct Image Loaders, CEGUI and BRDF Explorer GUI)
- ***[@manhnt9](https://github.com/manhnt9)*** **Nguyễn Tiến Mạnh** (CEGUI, Build System and Radeon Rays Proof-of-Concept Integration in Prime Engine X with IrrlichtBaW back-end)

#### Words of appreciation for developers whose software has been used in **Nabla**, currently and in the past:

- The initial Irrlicht 1.8.3 codebase
- **[OpenSSL](https://github.com/openssl/openssl)** and **[aesGladman](https://github.com/BrianGladman/aes)**
- **[SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)**  
- **[shaderc](https://github.com/google/shaderc)**
- **[zlib](https://github.com/madler/zlib)**, **[bzip](https://github.com/enthought/bzip2-1.0.6)**, **[libzip2](https://packages.debian.org/search?keywords=libzip2)**, **[lzma](https://github.com/jljusten/LZMA-SDK)** and **[lz4](https://github.com/lz4/lz4)**
- **[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)**, **[jpeglib](http://libjpeg.sourceforge.net/)** (past), **[libpng](https://github.com/glennrp/libpng)**
- Unicode convert_utf and utf8cpp (will be removed soon!)
