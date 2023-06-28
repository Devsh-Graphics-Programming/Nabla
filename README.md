# Nabla

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Nabla** (previously called **[IrrlichtBaW](https://github.com/buildaworldnet/IrrlichtBAW)** ) is a new renovated version of older **[Irrlicht](http://irrlicht.sourceforge.net/)** engine. 
The name change to Nabla allows for using Nabla side by side with the legacy Irrlicht and IrrlichtBaW engines. 
The project currently aims for a thread-able and *Vulkan*-centered API, the Vulkan backend is almost complete, and OpenGL and ES backends are currently in maintenance mode. 

This framework has been kindly begun by the founder ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** of **[Devsh Graphics Programming Sp. z O.O.](http://devsh.eu/)**  and was almost entirely sponsored by **Build A World Aps** in its early days, now it has been picked up by **[Ditt B.V.](https://www.ditt.nl/)**.

## (Get Hired) Jobs and Internships

If you are a programmer with a passion for High Performance Computing, Mathematics and Computer Graphics

If you can be in charge of your own time managment and work 4-day work weeks 100% remotely

Then make something impressive using Nabla, open a PR and contact us (jobs devsh.eu) with your CV.

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

Our language of choice is C++17 with C++11 and C11 coming in close second, however we're also amenable to C#, Java, Python and related languages.

Contact `newclients@devsh.eu` with inquires into contracting.

## Showcase

### Screenshots

### <u>Our Production Mitsuba Compatible Path Tracer made for Ditt B.V.</u>

Currently working on the `ditt` branch, in the process of being ported to Vulkan KHR Raytracing.

You can download a stable build [here](https://artifactory.devsh.eu/Ditt/ci/data/artifacts/public/Ditt.tar.bz2)

![](https://raw.githubusercontent.com/Devsh-Graphics-Programming/Nabla/17d76d294c46b4f3f3b98e9c4c6fd37c8396e502/site_media/readme/screenshots/4e86cf2b-3f8e-40eb-9835-6553ea205df2.jpg)

### [Multiple Importance Sampling and Depth of Field](https://www.youtube.com/watch?v=BuyVlQPV7Ks)

![](https://github.com/Devsh-Graphics-Programming/Nabla/blob/ad02c69e384c6655951c99d3b4bee5178a9dab2f/site_media/readme/gifs/myballs/Multiple%20Importance%20Sampling%20and%20Depth%20of%20Field%203.gif?raw=true)

![](https://github.com/Devsh-Graphics-Programming/Nabla/blob/ad02c69e384c6655951c99d3b4bee5178a9dab2f/site_media/readme/gifs/myballs/Multiple%20Importance%20Sampling%20and%20Depth%20of%20Field%205.gif?raw=true)

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

## Required Build Tools and SDK's

### Vanilla Build - most extensions disabled

- **[CMake](https://cmake.org/download/)** 
- **[MSVC](https://visualstudio.microsoft.com/pl/downloads/)** or **[GCC](https://sourceforge.net/projects/mingw-w64/)** or **[NDK's Clang](https://TODO.todo/)** 
- **[Vulkan SDK](https://vulkan.lunarg.com/sdk/home)**, at least **1.2.198.1** version ***without* any components** (they break out SPIR-V Tools integration) installed
- **[Perl 5.28 executable version](https://www.perl.org/get.html)**
- **[NASM](https://www.nasm.us/pub/nasm/releasebuilds/?C=M;O=D)**
- **[Python 3.8](https://www.python.org/downloads/release/python-380/)** or later (3.10.2 required for Renderdoc based GPU Automated Tests)

#### Vanilla + CUDA Build

**Nabla** only supports *CUDA* interop using the Driver API not the Runtime API. We use NVRTC to produce runtime compiled PTX.

Because *CUDA* needs its own version the GeForce (Quadro or Tesla) Driver, its often a few minor versions behind your automatically updated Windows driver, the install will fail even if it prompts you to agree to installing an older driver. 

So basically first remove your driver, then install *CUDA SDK*.

- **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**

#### CMake warnings in reference CUDA and notes

On Windows *CMake* has trouble finding new packages installed after *CMake*, so its the perfect time to visit **[its website](https://cmake.org/)** and check for a new version installer after installing *CUDA SDK*.

You can also thank NVidia for making the CUDA SDK a whole whopping 2.5 GB on Windows.

#### Vanilla + CUDA + Optix Build

After dealing with *CUDA* installing just install *Optix SKD*.

- **[OptiX SDK](https://developer.nvidia.com/designworks/optix/download)** 

### Android Build

**Required:**

- **[Android Studio](https://developer.android.com/studio)**
- **[JDK 8](https://www.java.com/download/)**

The first step is to install Android Studio and JDK 8. When done, open Android Studio and navigate to **Tools** -> **SDK Manager** -> **System Settings** -> **Android SDK**.
Select *SDK Platforms* and install proper individual SDK components - install Android version with Android API level you will be targeting. Then switch to *SDK Tools* and make sure to install **Android SDK Build-Tools 32** and **NDK (Side by side)** - it's a *requirement*! Also you must make sure that your **JAVA_HOME** enviroment variable is set to proper JDK installation path.

Now you can begin CMake'ing. We use **Ninja** generator tools as a generator for building Nabla for Android on both Windows and Linux Host-OSes. *Note that Nabla Android build has been performed and tested so far on Windows as cross compile OS with **Ninja** generator and on Linux as cross compile OS with **Makefile** and **Ninja** generators, but we recommend using **Ninja** for both OSes.* 

Before configuring CMake you must add 2 cache variables:

- **ANDROID_PLATFORM**
- **ANDROID_ABI**

**ANDROID_PLATFORM** is a target API platform that you pass as `android-x` where `x` is your android API level (you can pass 28 for instance). **ANDROID_ABI** is Application Binary Interface and note, that we support only `x86_64` currently. Those 2 cache variables *must be* specified before CMake configuration. Having done it you can specify toolchain file for cross-compiling by passing path to `android.toolchain.cmake`. You can find it in Android Studio's SDK directory in `ndk/<version>/build/cmake/android.toolchain.cmake`. Basically the entire path should look like this one `C:/Users/<your_user>/AppData/Local/AndroidSdk/ndk/<version>/build/cmake/android.toolchain.cmake`. With all of this feel free to generate.

Having Nabla generated you need to enter build directory, launch the terminal and type `cmake --build . --target Nabla -j4 -v` or if you want build android sample example you would type `cmake --build . --target android_sample_apk -j4 -v`. The android sample example produces *.apk* file you can use for debugging and profiling.

**Note:** each example provided by the engine builds as an executable with non-cross builds and with target of a name called `a_target`, in following example above it would be `android_sample`. When building cross-compile for android **to produce the APK file you need to add `_apk` postfix to the `a_target`, because `a_target` gets built then as a library.

#### DEPRECATED: Chrome Book SDK version

In order for the chromebook to work with the apk you build you need to install the right SDK version. Go to **Tools** -> **SDK Manager** -> **System Settings** -> **Android SDK** then select the *SDK Platforms* tab and tick the "Show Packake Details" checkbox in the bottom-right corner. After that select *Android 9.0 (Pie) -> Android SDK Platform 28* and hit "OK".

#### DEPRECATED: Chrome Book upload

To upload generated *.apk* into your ChromeBook you need first to make sure your device is in *developer mode* state. If it is, you can open Android Studio and choose Debug or Profile choosing *.apk* file. Then you will need to connect to your device using **adb** connector. To make use of adb, you need to find path to the executable that is placed in `C:/Users/<your_user>/AppData/Local/AndroidSdk/platform-tools` directory. When found, you can type in Android Studio command line `C:/Users/<your_user>/AppData/Local/AndroidSdk/platform-tools/adb connect <IP of ChromeBook network>`. You can find ChromeBook's IP by entering network settings and choosing current network ChromeBook is connected to. This way the ChromeBook should be listed in available devices and you should be able to upload *.apk* to the machine through debugging app shortcut. Take into account that you won't probably be able to debug in that scenario, but you will be able to upload *.apk* to the device.

#### DEPRECATED: Chrome Book debug

To debug the *.apk* on your chromebook you need to open the source file you want to debug in Android Studio (Either via *File->Open* or Drag&Drop, but be aware that d&d can deadlock your Android Studio 25% of the time so youll need to restart it), then place your breakpoints and hit "Debug" (The bug icon)  in the top right corner.

## External Dependencies

- **gl.h** header for *OpenGL* (possible to obtain even on headless servers from *mesa-libgl-devel*)

## Future Boost Library Dependencies

**Nabla** uses or will use the following **[*Boost*](https://www.boost.org/)** libraries:

- [ ] **[Bimap](http://man.hubwiz.com/docset/Boost.docset/Contents/Resources/Documents/boost/libs/bimap/doc/html/boost_bimap/bimap_and_boost.html)**
- [ ] **[Context](https://www.boost.org/doc/libs/1_61_0/libs/context/doc/html/context/overview.html)** (maybe, and if yes only the *fcontext_t* variant)
- [ ] **[Interprocess](https://www.boost.org/doc/libs/1_72_0/doc/html/interprocess.html)**
- [ ] **[Special](https://www.boost.org/doc/libs/1_72_0/libs/math/doc/html/special.html)**
- [ ] **[Stacktrace](https://www.boost.org/doc/libs/1_65_1/doc/html/stacktrace.html)** (we already have stack tracing on Linux, might be less work to do it on Windows)

The maybe's depend on how *xplatform* and easy to operate the *boost::context* is, esp w.r.t. Linux, Windows and Android. We will not use *boost::fibers* as we need our own complex scheduler.

## Building the Nabla library

### Cloning the project

**NOTICE: Due to GitHub SSH policy, our CI needed all submodules to be added with SSH URLs. THIS MEANS YOU NEED TO CHECKOUT THE SUPERPROJECT VIA SSH!**

Begin with cloning **Nabla** with:

```shell
git clone --recurse-submodules -j8 git@github.com:Devsh-Graphics-Programming/Nabla.git
```

If you haven't cloned `recursive`ly, you have to also perform:

```shell
git submodule init
git submodule update
```

*CMake* config script will try to initialize submodules for you however as well, but it doesn't mean the initialization attempt will be successful (it often is not when performed on a shaky internet connection, and you end up with dirty, locked or unversioned submodules).

### Submodules

If you haven't initialized the submodules yourself before the *CMake* configure step, and out *CMake* submodule update script destroyed them (badly/half initialized), you can run the following set of commands, but **beware** - it will completely wipe any changes to submodules.

```shell
git submodule foreach --recursive git clean -xfd
git submodule foreach --recursive git reset --hard
git submodule update --init --recursive
```

#### TODO: DOCUMENT THE NBL_UPDATE_SUBMODULE flag

By default Nabla's cmake...

But if you're working on making changes to one of our customized dependencies, you want to disable that, to not have the submodule reset on every CMake reconfigure (which may happen during a build).

#### Weird CMake behaviour, notes

Sometimes it may appear that there **won't be any files in submodules directories**. If so, you have to bring them back by using:

```shell
git reset --hard
```

on each submodule's directory required!

### CMake notes

#### Consider CMake and Visual Studio version, **important**! 

- The paragraph concerns *Visual Studio* only

Make sure you have installed the latest version of *Visual Studio* and *CMake*. Within older versions sometimes there may occur that *Visual Studio* outputs an error associated with compiler heap space. If you don't get any error, just skip the point. It's because having *x64* project opened the solution still uses *32 bit compiler exe* and cannot allocate **more than 4G of memory**, therefore **Nabla** is unbuildable. Furthermore *Visual Studio* doesn't provide any option to change that. Because of that you have to manually modify **.vcxproj xml** and add `x64` to `PropertyGroup` nodes. Pay attention that *CMake* generates a standard `PropertyGroup` node, but it isn't enough, because you need to put it into the target where building type is directly specified. It should look for instance as following:

```xml
<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <PreferredToolArchitecture>x64</PreferredToolArchitecture>
    <ConfigurationType>StaticLibrary</ConfigurationType>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>v142</PlatformToolset>
</PropertyGroup>
```

Current example above shows *Release* mode, but you should consider *Debug* as well. Having it done, 64bit cl.exe binary usage will be assured. When u get this problem, don't bother with editing all the **.vcxproj**s. It will probably only matter while building the engine, so the only thing you have to do is edit that **.vcxproj** you actually use - **Nabla.vcxproj** for instance and that's it. 

If you know a way to make *CMake* generating **.vcxproj**s already having those changes that will solve the issue, it will be great if you let us know about it :)

#### CMake on 64bit Windows

- Best to use *cmake-gui*

Run *cmake-gui* and, as usually, give *CMake* root **Nabla** directory and where you want *CMake* to put project files and click "Configure" field. When *CMake* asks you to choose compiler/IDE, make sure to check whether there's a distinct option for *64bit* mode and, if this is the case, choose this one.

For single-config IDEs (Code::Blocks) you'll have to manually set `CMAKE_BUILD_TYPE` to `Debug` or `Release`. `Release` is default.

You also have options `BUILD_EXAMPLES` and `BUILD_TOOLS` which do exactly what they say. By "*tools*" you should currently understand just `convert2baw`.

For Windows *MSVC* required, *MinGW* build system maintenance will be delegated to the community.

#### CMake on 64bit Linux

Same as Windows, except that currently we have no way of setting the correct working directory for executing the examples from within the IDE (for debugging). If you care about this please submit an *issue/PR/MR* to **[*CMake's* gitlab](https://gitlab.kitware.com/cmake/cmake)**.

We recommend the ***[Codelite IDE](https://codelite.org/)*** as that has a *CMake-gui* generator and has been tested and works relatively nice.

**[*Visual Studio Code*](https://code.visualstudio.com/)** suffers from a number of issues such as configuring the *CMake* every time you want to build a target and slow build times. Here are the issues:

1. **https://github.com/microsoft/vscode-cmake-tools/issues/771**
2. **https://github.com/microsoft/vscode-cmake-tools/issues/772**
3. **https://github.com/microsoft/vscode-cmake-tools/issues/773**

***[Clang](https://clang.llvm.org/) toolset*** is unmaintained and untested on Linux.

## First examples launching, significant notes

Remember you have to set up **starting target project** in *Visual Studio* before you begin to launch your example. To do that click on **Solution Explorer**, find the example name, hover on it and click on **Set as StartUp Project**. You can disable building examples by `NBL_BUILD_EXAMPLES` option in *CMake*.

## Use Nabla in your project!

To get **Nabla** to be used by an external application *without adding it as a subdirectory*,but still using a submodule, you should perform following:

```cmake
set(NBL_SOURCE_DIR "<YOUR_NABLA_SOURCE_DIRECTORY>") # PAY ATTENTION: you have to fill this one with Nabla source directory
set(NBL_BINARY_DIR "${NBL_SOURCE_DIR}/build")
set(NBL_INSTALL_DIR "${NBL_BINARY_DIR}/install")

list(APPEND NBL_CMAKE_ARGS "-DNBL_BUILD_DOCS:BOOL=OFF") # enable only if you have doxygen installed and detectable by cmake
list(APPEND NBL_CMAKE_ARGS "-DNBL_BUILD_EXAMPLES:BOOL=OFF")
list(APPEND NBL_CMAKE_ARGS "-DNBL_BUILD_TOOLS:BOOL=OFF") # the tools don't work yet (Apr 2020 status, might have changed since then)
list(APPEND NBL_CMAKE_ARGS "-DNBL_BUILD_MITSUBA_LOADER:BOOL=OFF") # you probably don't want this extension
list(APPEND NBL_CMAKE_ARGS "-D_NBL_COMPILE_WITH_BAW_LOADER_:BOOL=OFF") # you probably don't want this extension
list(APPEND NBL_CMAKE_ARGS "-D_NBL_COMPILE_WITH_BAW_WRITER_:BOOL=OFF") # you probably don't want this extension
list(APPEND NBL_CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${NBL_INSTALL_DIR}") # because of https://gitlab.kitware.com/cmake/cmake/-/issues/18790

ExternalProject_Add(Nabla
    DOWNLOAD_COMMAND  ""
    SOURCE_DIR        "${NBL_SOURCE_DIR}"
    BINARY_DIR        "${NBL_BINARY_DIR}"
    CMAKE_ARGS        ${NBL_CMAKE_ARGS}
    TEST_COMMAND      ""
)

include(${NBL_SOURCE_DIR}/cmake/build/AddNablaModule.cmake)

# now if you create executable you can use addNablaModule
add_executable(executableTest main.cpp) # assuming main.cpp exsists

# add Nabla module to "executableTest"
addNablaModule(executableTest "${NBL_INSTALL_DIR}")
```

If you want to use git (without a submodule) then you can use `ExternalProject_Add` with the `GIT_` properties instead.

I recommend you use `ExternalProject_Add` instead of `add_subdirectory` for **Nabla** as we haven't  tested its use by *3rdparty* applications that use *CMake* to build themselves yet.

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

## Automated Builds (TODO)

## License

**Nabla** is released under the **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)** license. See [**LICENSE.md**](https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/LICENSE.md) for more details.

## Documentation (WIP/TODO)

## Official Support (Discord)

Permament members of *Devsh Graphics Programming Sp. z O.O.* use this to organise publicly visible work. **[Join to the server](https://discord.gg/4MTCVaN)** to get into more details.

## Credits and Attribution

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
t
