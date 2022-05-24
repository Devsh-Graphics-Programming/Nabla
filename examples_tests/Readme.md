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
| 02.ComputeShader                | Y            | Y                | Y            | B            | B               | S            | B                 | S              |                                                   |
| 03.GPU_Mesh                     | W            | W                | W            | W            | W               | W            | W                 | W              |                                                   |
| 04.Keyframe                     | S            | S                | S            | S            | S               | S            | S                 | S              |                                                   |
| 05.NablaTutorialExample         | Y            | Y                | Y            | S            | S               | S            | S                 | S              |                                                   |
| 06.MeshLoaders                  | Y            | Y                | Y            | S            | S               | S            | Y                 | Y              |                                                   |
| 07.SubpassBaking                | Y            | Y                | Y            | S            | S               | S            | S                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 09.ColorSpaceTest               | B            | B                | B            | W            | W               | W            | W                 | W              |                                                   |
| 10.AllocatorTest                | Y            | Y                | Y            | S            | S               | S            | N                 | N              |                                                   |
| 11.LoDSystem                    | Y            | Y                | B            | S            | N               | S            | N                 | S              |                                                   |
| 12.glTF                         | W            | N                | W            | W            | N               | W            | N                 | W              | COMPILE_WITH_GLTF_LOADER                          |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 14.ComputeScan                  | Y            | Y                | B            | S            | S               | S            | S                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 16.OrderIndependentTransparency | B            | B                | B            | S            | S               | S            | S                 | S              |                                                   |
| 17.SimpleBulletIntegration      | B            | N                | B            | S            | N               | S            | N                 | N              | BUILD_BULLET                                      |
| 18.MitsubaLoader                | S            | N                | S            | S            | N               | S            | N                 | N              | BUILD_MITSUBA_LOADER                              |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 20.Megatexture                  | W            | W                | W            | S            | S               | S            | N                 | S              |                                                   |
| 21.DynamicTextureIndexing       | B            | B                | B            | S            | N               | S            | N                 | S              |                                                   |
| 22.RaytracedAO                  | N            | N                | W            | N            | N               | W            | N                 | N              | BUILD_MITSUBA_LOADER                              |
| 23.Autoexposure                 | Y            | Y                | Y            | S            | S               | S            | N                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 25.Blur                         | S            | N                | S            | S            | N               | S            | N                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 27.PLYSTLDemo                   | Y            | Y                | B            | S            | S               | S            | N                 | N              | COMPILE_WITH_STL_LOADER & COMPILE_WITH_PLY_LOADER |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 29.SpecializationConstants      | B            | B                | Y            | S            | S               | S            | N                 | S              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 33.Draw3DLine                   | Y            | Y                | Y            | S            | S               | S            | S                 | S              |                                                   |
| 34.LRUCacheUnitTest             | Y            | Y                | Y            | Y            | Y               | Y            | N                 | N              |                                                   |
| 35.GeometryCreator              | Y            | Y                | Y            | S            | S               | S            | N                 | S              |                                                   |
| 36.CUDAInterop                  | N            | N                | W            | N            | N               | W            | N                 | N              | COMPILE_WITH_CUDA                                 |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 38.EXRSplit                     | S            | S                | S            | S            | S               | S            | N                 | N              |                                                   |
| 39.DenoiserTonemapper           | N            | N                | W            | N            | N               | W            | N                 | N              | COMPILE_WITH_CUDA & COMPILE_WITH_OPTIX            |
| 40.GLITest                      | S            | S                | S            | S            | S               | S            | N                 | S              | COMPILE_WITH_GLI_LOADER                           |
| 41.VisibilityBuffer             | S            | S                | S            | S            | S               | S            | N                 | N              |                                                   |
| 42.FragmentShaderPathTracer     | B            | B                | Y            | S            | S               | S            | S                 | S              |                                                   |
| 43.SumAndCDFFilters             | Y            | Y                | Y            | S            | N               | S            | N                 | N              |                                                   |
| 44.LevelCurveExtraction         | S            | N                | S            | S            | N               | S            | N                 | N              |                                                   |
| 45.BRDFEvalTest                 | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 46.SamplingValidation           | S            | S                | S            | S            | S               | S            | N                 | S              |                                                   |
| 47.DerivMapTest                 | Y            | Y                | B            | S            | N               | S            | N                 | N              |                                                   |
| 48.ArithmeticUnitTest           | Y            | B                | B            | S            | S               | S            | N                 | S              |                                                   |
| 49.ComputeFFT                   | S            | N                | S            | S            | N               | S            | N                 | N              |                                                   |
| 50.NewAPITest                   | W            | W                | W            | W            | W               | W            | W                 | W              |                                                   |
| 51.RadixSort                    | W            | N                | W            | W            | N               | W            | N                 | W              |                                                   |
| 52.SystemTest                   | Y            | Y                | Y            | S            | S               | S            | S                 | S              |                                                   |
| 53.ComputeShaders               | B            | B                | B            | S            | N               | S            | N                 | S              |                                                   |
| 54.Transformations              | Y            | Y                | B            | S            | S               | S            | S                 | S              |                                                   |
| 55.RGB18E7S3                    | Y            | Y                | Y            | S            | S               | S            | N                 | N              |                                                   |
| 56.RayQuery                     | N            | N                | Y            | N            | N               | S            | N                 | S              |                                                   |
| 57.AndroidSample                | N            | N                | N            | N            | N               | N            | S                 | S              |                                                   |
| 58.MediaUnpackingOnAndroid      | N            | N                | N            | N            | N               | N            | Y                 | Y              |                                                   |
| FREE_SLOT                       |              |                  |              |              |                 |              |                   |                |                                                   |
| 60.ClusteredRendering           | W            | N                | W            | W            | N               | W            | N                 | N              |                                                   |

`*` Only Nvidia provides a working GLES 3.1 driver with OES_texture_view on Windows, so we only test there.

`**` Needs the Xcb implementation of the `ui::` namespace to be complete.

`***` Only x86_64 architecture supported for Android builds, also NBL_BUILD_ANDROID is required.

`****` NBL_BUILD_EXAMPLES is needed for any example to build!
