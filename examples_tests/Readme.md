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

_Examples numbered 00 are provisional and are not part of the example suite._

Y = Already Works
B = Has a known bug
W = Work-In-Progress, sample logic not complete or temporarily modifed
S = Intended to be Supported (requires some work to port after an API change)
N = No support
|                                 | Win32 OpenGL | Win32 OpenGL ES* | Win32 Vulkan | X11** OpenGL | X11** OpenGL ES | X11** Vulkan | Android OpenGL ES | Android Vulkan | Required CMake Options****                        |
|---------------------------------|--------------|------------------|--------------|--------------|-----------------|--------------|-------------------|----------------|---------------------------------------------------|
| 01.HelloWorld                   | Y            | Y                | Y            | S            | S               | S            | Y                 | S              |                                                   |
| 02.ComputeShader                | B            | B                | Y            | B            | B               | S            | B                 | S              |                                                   |
| 03.GPU_Mesh                     | S            | S                | S            | S            | S               | S            | S                 | S              |                                                   |
| 04.Keyframe                     | S            | S                | S            | S            | S               | S            | S                 | S              |                                                   |
| 05.NablaTutorialExample         | Y            | Y                | Y            | S            | S               | S            | S                 | S              |                                                   |
| 06.MeshLoaders                  | Y            | Y                | Y            | S            | S               | S            | Y                 | Y              |                                                   |
| 07.SubpassBaking                | S            | S                | S            | S            | S               | S            | S                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 09.ColorSpaceTest               | W            | W                | W            | W            | W               | W            | W                 | W              |                                                   |
| 10.AllocatorTest                | Y            | Y                | Y            | S            | S               | S            | N                 | N              |                                                   |
| 11.LoDSystem                    | Y            | S                | B            | S            | S               | S            | N                 | S              |                                                   |
| 12.glTF                         | W            | W                | W            | W            | W               | W            | W                 | W              | COMPILE_WITH_GLTF_LOADER                          |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 14.ComputeScan                  | B            | B                | B            | S            | S               | S            | N                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 16.OrderIndependentTransparency | B            | B                | B            | S            | S               | S            | S                 | S              |                                                   |
| 17.SimpleBulletIntegration      | Y            | Y                | Y            | S            | S               | S            | N                 | N              | BUILD_BULLET                                      |
| 18.MitsubaLoader                | S            | S                | S            | S            | S               | S            | N                 | N              | BUILD_MITSUBA_LOADER                              |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 20.Megatexture                  | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 21.DynamicTextureIndexing       | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 22.RaytracedAO                  | W            | W                | W            | W            | W               | W            | W                 | W              | BUILD_MITSUBA_LOADER                              |
| 23.Autoexposure                 |              |                  |              |              |                 |              |                   |                |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 25.Blur                         |              |                  |              |              |                 |              |                   |                |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 27.PLYSTLDemo                   |              |                  |              |              |                 |              |                   |                | COMPILE_WITH_STL_LOADER & COMPILE_WITH_PLY_LOADER |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 29.SpecializationConstants      | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 33.Draw3DLine                   | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 34.LRUCacheUnitTest             | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 35.GeometryCreator              | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 36.CUDAInterop                  | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 38.EXRSplit                     | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 39.DenoiserTonemapper           | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 40.GLITest                      | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 41.VisibilityBuffer             | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 42.FragmentShaderPathTracer     | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 43.SumAndCDFFilters             | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 44.LevelCurveExtraction         | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 45.BRDFEvalTest                 | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 46.SamplingValidation           | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 47.DerivMapTest                 | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 48.ArithmeticUnitTest           | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 49.ComputeFFT                   | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 50.ArithmeticUnitTest           | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 51.RadixSort                    | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 52.SystemTest                   | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 53.ComputeShaders               | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 54.Transformations              | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 55.RGB18E7S3                    | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 56.RayQuery                     | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 57.AndroidSample                | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 58.MediaUnpackingOnAndroid      | N            | N                | N            | N            | N               | N            | Y                 | Y              | None                                              |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 60.ClusteredRendering           | W            | W                | W            | W            | W               | W            | N                 | N              |                                                   |

* Only Nvidia provides a working GLES 3.1 driver with OES_texture_view on Windows, so we only test there.
** Needs the Xcb implementation of the `ui::` namespace to be complete.
*** Only x86_64 architecture supported for Android builds, also NBL_BUILD_ANDROID is required.
**** NBL_BUILD_EXAMPLES is needed for any example to build!
