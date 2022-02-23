# Examples and Tests

The Nabla Examples are documentation-by-example and building blocks for our future Continuous Integration GPU Integration Tests.

## Where can I find the makefiles or IDE projects/solutions?

Given an example in folder `XY.ExampleName`, CMake will generate either a target or a separate makefile/project/solution called `examplename` (no number, always lowercase).

Whenever CMake generates separate makefiles/solutions/projects, they will be generated in the `./examples_tests` under the build directory you supplied to CMake.

**Samples are meant to be built into the `./bin` directory in the source (its git-ignored) and invoked with that Current Working Directory.**

**WARNING:** If you're using an IDE different than Visual Studio you need to set the CWD correctly for when you start the example for Debugging!

**WARNING:** Only generation of IDE projects by standalone CMake is supported, we do not use or rely on IDE integrations of CMake.

## Maintenance Matrix

In the future we expect this matrix to be kept up to date, live by our CI.

Y = Already Works
B = Has a known bug
W = Work-In-Progress, sample logic not complete or temporarily modifed
S = Intended to be Supported (requires some work to port after an API change)
N = No support
|                                 | Win32 OpenGL | Win32 OpenGL ES* | Win32 Vulkan | X11** OpenGL | X11** OpenGL ES | X11** Vulkan | Android OpenGL ES | Android Vulkan | Required NBL_* CMake Options |
|---------------------------------|--------------|------------------|--------------|--------------|-----------------|--------------|-------------------|----------------|------------------------------|
| 01.HelloWorld                   | Y            | Y                | Y            | S            | S               | S            | Y                 | S              | None                         |
| 02.ComputeShader                | B            | B                | Y            | S            | S               | S            | Y                 | S              | None                         |
| 03.GPU_Mesh                     |              |                  |              |              |                 |              |                   |                |                              |
| 04.Keyframe                     |              |                  |              |              |                 |              |                   |                |                              |
| 05.NablaTutorialExample         |              |                  |              |              |                 |              |                   |                |                              |
| 06.MeshLoaders                  |              |                  |              |              |                 |              |                   |                |                              |
| 07.SubpassBaking                |              |                  |              |              |                 |              |                   |                |                              |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                              |
| 09.ColorSpaceTest               |              |                  |              |              |                 |              |                   |                |                              |
| 10.AllocatorTest                |              |                  |              |              |                 |              |                   |                |                              |
| 11.LoDSystem                    |              |                  |              |              |                 |              |                   |                |                              |
| 12.glTF                         |              |                  |              |              |                 |              |                   |                |                              |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                              |
| 14.ComputeScan                  |              |                  |              |              |                 |              |                   |                |                              |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                              |
| 16.OrderIndependentTransparency |              |                  |              |              |                 |              |                   |                |                              |
| 17.SimpleBulletIntegration      |              |                  |              |              |                 |              |                   |                |                              |
| 18.MitsubaLoader                |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
|                                 |              |                  |              |              |                 |              |                   |                |                              |
| 58.MediaUnpackingOnAndroid      | N            | N                | N            | N            | N               | N            | Y                 | Y              | None                         |

* Only Nvidia provides a working GLES 3.1 driver with OES_texture_view on Windows, so we only test there.
