# IrrlichtBAW

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**IrrlichtBAW** is a new renovated version of older **[Irrlicht](http://irrlicht.sourceforge.net/)** engine. The project currently aims for a thread-able and *Vulkan*-centered API, but currently works on *OpenGL* only. This framework has been kindly begun by the founder ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** of Devsh Graphics Programming Sp. z O.O. and almost entirely sponsored by **Build A World Aps**. in it's early days, and now picked up by the **[Ditt](https://www.ditt.nl/)** company. The `stable-ish` branch is used in production releases of **[Build A World EDU](https://edu.buildaworld.net/)**, since 2015. The framework has been used both for game development and ArchViz.

## Contracting

The members of Devsh Graphics Programming Sp. z O.O. (Company Registration (KRS) #: 0000764661) are available (individually or collectively) for contracts on projects of various scopes and timescales, especially on foreign frameworks, codebases and third-party 3D frameworks. We provide expertise in OpenGL, OpenGL ES, WebGL, Vulkan, OpenCL, CUDA, D3D12 and D3D11, computer vision, Audio programming, DSP, video encoding and decoding as well as more generalized High Performance Computing. Our language of choice is C++17 with C++11 and C11 coming in close second, however we're also amenable to Java, Python and related languages.

Contact ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** (e-mail available in the GitHub profile) with inquires into contracting.

## Showcase

### Screen Shots

<u>**BRDF Explorer**</u>

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/BRDF%20Explorer.png?raw=true)

<u>**Many light raytracing**</u>

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Many%20light%20raytracing.png?raw=true)

<u>**Many light raytracing**</u>

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Many%20light%20raytracing%202.png?raw=true)

**<u>Cylindrical light source</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Cylindrical%20light%20source.png?raw=true)

**<u>Two area lights</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Two%20area%20lights.png?raw=true)

**<u>.OBJ Loader with MTL pipeline integration</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/.OBJ%20Loader%20with%20MTL%20pipeline%20integration.png?raw=true)

**<u>Over 10 area lights</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Over%2010%20area%20lights.png?raw=true)

**<u>Raytracing sample</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Raytracing%20sample.png?raw=true)

**<u>Raytracing sample</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Raytracing%20sample%202.png?raw=true)

**<u>Light emitters</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Light%20emitters.png?raw=true)

**<u>Raytracing sample</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/Raytracing%20sample%203.png?raw=true)

**<u>1 Megapixel, 1 Million Samples</u>**

![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/screenshots/1%20Megapixel,%201%20Million%20Samples.png?raw=true)



### Gifs - [Raytracing flythrough](https://www.youtube.com/watch?v=bwVVoAsRjHI)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/raytracingflythrough/raytracing%201.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/raytracingflythrough/raytracing%202.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/raytracingflythrough/raytracing%203.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/raytracingflythrough/raytracing%204.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/raytracingflythrough/raytracing%205.gif?raw=true)



### Gifs - [Raytracing With Optix AI Denoising [Albedo and Normals]](https://www.youtube.com/watch?v=VFad-Y-dSxQ&feature=youtu.be)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/denoisingalbedoandnormals/denoising%201.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/denoisingalbedoandnormals/denoising%202.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/denoisingalbedoandnormals/denoising%203.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/denoisingalbedoandnormals/denoising%204.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/denoisingalbedoandnormals/denoising%205.gif?raw=true)



![](https://github.com/buildaworldnet/IrrlichtBAW/blob/6e4842588ef02ffb619242a08c0c037bba59c244/site_media/readme/gifs/denoisingalbedoandnormals/denoising%207.gif?raw=true)



## Main Features

- **Asset management pipeline**
- **Automatic pipeline layout creation**
- **Shader introspection**
- **Using SPIR-V shaders in OpenGL**
- **Libraries of GLSL shader functions**
- **Compute shaders**
- **Virtual Texturing**
- **CUDA and OpenGL interop**
- **OpenCL and OpenGL interop**
- **CPU image filtering**

## Main Delivered Extensions

- **Mitsuba scene loader (auto-generated shaders)** 
- **Tonemapper**
- **Fastest blur on the planet**
- **Bullet physics beginner integration** 
- **OptiX interop**
- **Radeon rays interop**

## Platforms

- [x] **Windows**

- [x] **Linux**

- [ ] **Android 7.0 +**

- [ ] **Mac OS**

- [ ] **iOS**


## Required Build Tools and SDK's

### Vanilla Build - most extensions disabled

- **[CMake](https://cmake.org/download/)** 
- **[MSVC](https://visualstudio.microsoft.com/pl/downloads/)** or **[GCC](https://sourceforge.net/projects/mingw-w64/)**
- **[Vulkan SDK](https://vulkan.lunarg.com/sdk/home)**
- **[Perl](https://www.perl.org/get.html)**
- **[NASM](https://www.nasm.us/pub/nasm/releasebuilds/?C=M;O=D)**
- **[Python 2.7](https://www.python.org/download/releases/2.7/)** or later

### Vanilla + CUDA Build

**IrrlichtBAW** only supports *CUDA* interop using the Driver API not the Runtime API. We use the runtime compiled CUDA.

Because *CUDA* needs its own version the GeForce (Quadro or Tesla) Driver, its often a few minor versions behind your automatically updated Windows driver, the install will fail even if it prompts you to agree to installing an older driver. 

So basically first remove your driver, then install *CUDA SDK*.

- **[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**

#### CMake warnings in reference CUDA and notes

On Windows *CMake* has trouble finding new packages installed after *CMake*, so its the perfect time to visit **[it's website](https://cmake.org/)** and check for a new version installer after installing *CUDA SDK*.

You can also thank NVidia for making the CUDA SDK a whole whopping 2.5 GB on Windows.

### Vanilla + CUDA + Optix Build

After dealing with *CUDA* installing just install *Optix SKD*.

- **[OptiX SDK](https://developer.nvidia.com/designworks/optix/download)** 

## External Dependencies

- **gl.h** header for *OpenGL* (possible to obtain even on headless servers from *mesa-libgl-devel*)
- **OpenCL SDK** (can get rid of it by compiling without the two *OpenCL* files)

## Future Boost Library Dependencies

**IrrlichtBAW** uses or will use the following **[*Boost*](https://www.boost.org/)** libraries:

- [ ] **[Bimap](http://man.hubwiz.com/docset/Boost.docset/Contents/Resources/Documents/boost/libs/bimap/doc/html/boost_bimap/bimap_and_boost.html)**
- [ ] **[Context](https://www.boost.org/doc/libs/1_61_0/libs/context/doc/html/context/overview.html)** (maybe, and if yes only the *fcontext_t* variant)
- [ ] **[Interprocess](https://www.boost.org/doc/libs/1_72_0/doc/html/interprocess.html)**
- [ ] **[Special](https://www.boost.org/doc/libs/1_72_0/libs/math/doc/html/special.html)**
- [ ] **[Stacktrace](https://www.boost.org/doc/libs/1_65_1/doc/html/stacktrace.html)** (we already have stack tracing on Linux, might be less work to do it on Windows)

The maybe's depend on how *xplatform* and easy to operate the *boost::context* is, esp w.r.t. Linux, Windows and Android. We will not use *boost::fibers* as we need our own complex scheduler.

## Building IrrlichtBAW library

### Cloning the project

Begin with cloning **IrrlichtBAW** with:

```shell
git clone --recurse-submodules -j8 https://github.com/buildaworldnet/IrrlichtBAW.git
```

If you haven't cloned `recursive`ly, you have to also perform:

```shell
git submodule init
git submodule update
```

*CMake* config script will try to initialize submodules for you however as well, but it doesn't mean the initialization attempt will be successful.

### Submodules

If you haven't initialized the submodules yourself before the *CMake* configure step, and out *CMake* submodule update script destroyed them (badly/half initialized), you can run the following set of commands, but **beware** - it will completely wipe any changes to submodules.

```shell
git submodule foreach --recursive git clean -xfd
git submodule foreach --recursive git reset --hard
git submodule update --init --recursive
```

#### Weird CMake behaviour, notes

Sometimes it may appear that there **won't be any files in submodules directories**. If so, you have to bring them back by using:

```shell
git reset --hard
```

on each submodule's directory required! Furthermore you have to:

```shell
git checkout tags/glew-cmake-2.1.0
```

in *glew* directory that you can find in ***3rdparty/CEGUI/glew*** directory because of *glew* commiting politics. Having done it you can switch to your ***master/root*** directory and commit those changes if you want, but it isn't necessary to compile entire library.

### CMake notes

#### Consider CMake and Visual Studio version, **important**! 

- The paragraph concerns *Visual Studio* only

Make sure you have installed the latest version of *Visual Studio* and *CMake*. Within older versions sometimes there may occour that *Visual Studio* outputs an error associated with compiler heap space. If you don't get any error, just skip the point. It's because having *x64* project opened the solution still uses *32 bit compiler exe* and cannot allocate **more than 4G of memory**, therefore **IrrlichtBAW** is unbuildable. Furthermore *Visual Studio* doesn't provide any option to change that. Because of that you have to manually modify **.vcxproj xml** and add `x64` to `PropertyGroup` nodes. Pay attention that *CMake* generates a standard `PropertyGroup` node, but it isn't enough, because you need to put it into the target where building type is directly specified. It should look for instance as following:

```xml
<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Release|x64'" Label="Configuration">
    <PreferredToolArchitecture>x64</PreferredToolArchitecture>
    <ConfigurationType>StaticLibrary</ConfigurationType>
    <CharacterSet>MultiByte</CharacterSet>
    <PlatformToolset>v142</PlatformToolset>
</PropertyGroup>
```

Current example above shows *Release* mode, but you should consider *Debug* as well. Having it done, 64bit cl.exe binary usage will be assured. When u get this problem, don't bother with editing all the **.vcxproj**s. It will probably only matter while building the engine, so the only thing you have to do is edit that **.vcxproj** you actually use - **Irrlicht.vcxproj** for instance and that's it. 

If you know a way to make *CMake* generating **.vcxproj**s already having those changes that will solve the issue, it will be great if you let us know about it :)

#### CMake on 64bit Windows

- Best to use *cmake-gui*

Run *cmake-gui* and, as usually, give *CMake* root **IrrlichtBAW** directory and where you want *CMake* to put project files and click "Configure" field. When *CMake* asks you to choose compiler/IDE, make sure to check whether there's a distinct option for *64bit* mode and, if this is the case, choose this one.

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

Remember you have to set up **starting target project** in *Visual Studio* before you begin to launch your example. To do that click on **Solution Explorer**, find the example name, hover on it and click on **Set as StartUp Project**. You can disable building examples by `IRR_BUILD_EXAMPLES` option in *CMake*.

## Use IrrlichtBaW in your project!

To get **IrrlichtBaW** to be used by an external application *without adding it as a subdirectory*,but still using a submodule, you should perform following:

```cmake
list(APPEND IRR_CMAKE_ARGS "-DIRR_BUILD_DOCS:BOOL=OFF") # enable only if you have doxygen installed and detectable by cmake
list(APPEND IRR_CMAKE_ARGS "-DIRR_BUILD_EXAMPLES:BOOL=OFF")
list(APPEND IRR_CMAKE_ARGS "-DIRR_BUILD_TOOLS:BOOL=OFF") # the tools don't work yet (Apr 2020 status, might have changed since then)
list(APPEND IRR_CMAKE_ARGS "-DIRR_BUILD_MITSUBA_LOADER:BOOL=OFF") # you probably don't want this extension
ExternalProject_Add(IrrlichtBaW
    DOWNLOAD_COMMAND  ""
    SOURCE_DIR        "${IRR_SOURCE_DIR}"
    BINARY_DIR        "${IRR_BINARY_DIR}"
    INSTALL_DIR       "${IRR_INSTALL_DIR}"
    CMAKE_ARGS        ${IRR_CMAKE_ARGS}
    TEST_COMMAND      ""
)
```

 If you want to use git (without a submodule) then you can use `ExternalProject_Add` with the `GIT_` properties instead.

I recommend you use `ExternalProject_Add` instead of `add_subdirectory` for **IrrlichtBaW** as we haven't  tested its use by *3rdparty* applications that use *CMake* to build themselves yet (**BaW EDU** uses it directly from *MSVC*/*make* like it's still the stone-age of build systems).

## License

**IrrlichtBAW** is released under the **[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)** license. See [**LICENSE.md**](https://github.com/buildaworldnet/IrrlichtBAW/blob/master/LICENSE.md) for more details.

## API documentation, help and extra improvements

If you would like to take care of documenting some files, please **[click it](https://github.com/buildaworldnet/IrrlichtBAW/wiki/Documentation)**. If you feel like you'd be interesting in improving and maintaining this repository's wiki, contact ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** on **[Discord](https://discord.gg/4MTCVaN)**.

## Official Support (Discord)

Permament members of *Devsh Graphics Programming Sp. z O.O.* use this to organise publicly visible work. **[Join to the server](https://discord.gg/4MTCVaN)** to get into more details. There's also a skype support group, reach ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** for a private invite.

## Credits and Attribution

#### The authors of **IrrlichtBAW** are:

- **Mateusz Kielan** ***[@devshgraphicsprogramming](https://github.com/devshgraphicsprogramming)*** (Lead)
- **Krzysztof Szenk** ***[@crisspl](https://github.com/Crisspl)*** (Core Engineer)
- **Arkadiusz Lachowicz** ***[@AnastaZIuk](https://github.com/AnastaZIuk)*** (Junior Programmer)
- **Przemysław Pachytel** ***[@Przemog1](https://github.com/Przemog1)*** (Junior Programmer)
- **Cyprian Skrzypczak** ***[@Hazardu](https://github.com/Hazardu)*** (Junior Programmer)
- **Søren Gronbech** 

#### Past Authors and Contributors:

- ***[@khom-khun](https://github.com/khom-khun)*** (Bullet Physics Extension + Example and **[the irrBaW-test repository of easy to understand demos](https://github.com/khom-khun/irrBAW-test)**)
- ***[@manhnt9](https://github.com/manhnt9)*** **Nguyễn Tiến Mạnh** (CEGUI, Build System and Radeon Rays Proof-of-Concept Integration in Prime Engine X with IrrlichtBaW back-end)
- ***[@florastamine](https://github.com/florastamine)*** **Nguyễn Ngọc Huy** (sRGB-Correct Image Loaders, CEGUI and BRDF Explorer GUI)

#### Words of appreciation for developers whose software has been used in **IrrlichtBAW**, currently and in the past:

- The initial Irrlicht 1.8.3 codebase
- **[OpenSSL](https://github.com/openssl/openssl)** and **[aesGladman](https://github.com/BrianGladman/aes)**
- **[SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)**  
- **[shaderc](https://github.com/google/shaderc)**
- **[zlib](https://github.com/madler/zlib)**, **[bzip](https://github.com/enthought/bzip2-1.0.6)**, **[libzip2](https://packages.debian.org/search?keywords=libzip2)**, **[lzma](https://github.com/jljusten/LZMA-SDK)** and **[lz4](https://github.com/lz4/lz4)**
- **[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo)**, **[jpeglib](http://libjpeg.sourceforge.net/)** (past), **[libpng](https://github.com/glennrp/libpng)**
- Unicode convert_utf and utf8cpp
