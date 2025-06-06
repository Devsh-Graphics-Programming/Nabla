# NSC & Godbolt integration

## Run Compiler Explorer with NSC tool in docker container!

https://github.com/user-attachments/assets/8d409477-92e4-4238-b5e5-637cfbdf7263

<p align="center">
  <a href="https://github.com/Devsh-Graphics-Programming/Nabla/actions">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Devsh-Graphics-Programming/Nabla/badges/packages/nabla-shader-compiler-nsc/image-badge.json" alt="Image Status" /></a>
  <a href="https://github.com/Devsh-Graphics-Programming/Nabla/actions">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Devsh-Graphics-Programming/Nabla/badges/nabla/build.json" alt="Build Status" /></a>
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License: Apache 2.0" /></a>
  <a href="https://discord.gg/krsBcABm7u">
    <img src="https://img.shields.io/discord/308323056592486420?label=discord&logo=discord&logoColor=white&color=7289DA" alt="Join our Discord" /></a>
</p>

## Requirements

- Configured [***Docker***](https://docs.docker.com/desktop/setup/install/windows-install/) for Windows Containers
- [Windows, Windows Server Core or Windows Server](<https://learn.microsoft.com/en-us/virtualization/windowscontainers/manage-containers/container-base-images>) with **minumum** x86_64 10.0.20348 build (2022 distributions)

> [!TIP]
> type `cmd /ver` to see your build version

> [!WARNING]  
> You cannot run it on Windows Home Edition as it doesn't have `Containers` feature, visit Microsoft [docs](<https://learn.microsoft.com/en-gb/virtualization/windowscontainers/quick-start/set-up-environment?tabs=dockerce>) for more details

> [!CAUTION]  
> Hyper-V is **NOT** supported, you must run NSC Godbolt container as process

## How to run image

> [!IMPORTANT]  
> If using Docker Desktop - first make sure you have switched to `Containers for Windows`, see image bellow. If you are CLI user and have client & daemon headless then use appropriate windows build context.

![Containers for Windows](https://user-images.githubusercontent.com/65064509/152947300-affca592-35a7-4e4c-a7fc-2055ce1ba528.png)

> [!CAUTION]  
> Examples bellow use `docker compose` to run the image but if you want to `docker run` then make sure to mount required system directories and expose port otherwise it will fail in runtime, see the [compose](<https://github.com/Devsh-Graphics-Programming/Nabla/blob/master/compose.yml#L6>) file for more details

### from container registry

execute

```powershell
curl -L https://raw.githubusercontent.com/Devsh-Graphics-Programming/Nabla/master/compose.yml | docker compose -f - up
```

or in Nabla checkout

```powershell
docker compose up
```

and type `localhost` in your browser.

### from Nabla pipeline workflow artifacts

> [!NOTE]
> We publish container images to the GitHub Container Registry that include **only the Release variant** of NSC executables built with **MSVC**.  
> However, our CI pipelines **build and test all configurations**. Compressed images for each configuration are uploaded as **workflow artifacts**.
> Look for artifacts named:  
> `<prefix>-msvc-<config>-nsc-godbolt-image`

> [!NOTE]
> To decompress image artifact you need [zstd](<https://github.com/facebook/zstd/releases>)

Download workflow image artifact, unzip and

```powershell
zstd -d < <prefix>-msvc-<config>-nsc-godbolt-image.tar.zst | docker load
```

<details>
<summary>Docker load example (click to expand)</summary>

```  
C:\Users\anastaziuk\Desktop\DevshGraphicsProgramming\Nabla\tools\nsc\docker>zstd -d < run-windows-17.13.6-msvc-Debug-nsc-godbolt-image.tar.zst | docker load
b2ebf78c3627: Loading layer [==================================================>]  3.149MB/3.149MB
4c201e14cc01: Loading layer [==================================================>]   77.4MB/77.4MB
68a216251b8f: Loading layer [==================================================>]  61.95kB/61.95kB
7a4e13ca4c4e: Loading layer [==================================================>]  52.74kB/52.74kB
634001f55b21: Loading layer [==================================================>]  52.74kB/52.74kB
6a609178bb9a: Loading layer [==================================================>]  52.74kB/52.74kB
3d7afb042308: Loading layer [==================================================>]  52.74kB/52.74kB
ca034d7bc58a: Loading layer [==================================================>]  52.74kB/52.74kB
55b4134a1ae9: Loading layer [==================================================>]  52.74kB/52.74kB
0648adff3faa: Loading layer [==================================================>]  52.74kB/52.74kB
Loaded image: ghcr.io/devsh-graphics-programming/nabla:nsc-godbolt-build-msvc-debug-17.13.6
```

</details>

copy `compose.yml` in Nabla root directory to eg. `override-compose.yml`, replace it's `image` field value with loaded image name (eg. `ghcr.io/devsh-graphics-programming/nabla:nsc-godbolt-build-msvc-debug-17.13.6` like in the example) then execute

```
docker compose -f override-compose.yml up
```

and type `localhost` in your browser.

## How to build image

Configure CMake with `NBL_ENABLE_DOCKER_INTEGRATION` and build `run-compiler-explorer` target.
