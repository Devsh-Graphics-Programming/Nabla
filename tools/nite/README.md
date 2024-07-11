# Nabla IMGUI Test Engine

Performs suite tests for Nabla IMGUI backend.

## CTest

Build the target with desired configuration eg. `Debug`, open command line in the target's build directory and execute

```bash
ctest -C Debug --progress --stop-on-failure
```

<video src="https://github.com/Devsh-Graphics-Programming/Nabla-Site-Media/raw/master/tools/nite/nite-ctest.mp4"></video>

CTest will execute `NBL_NITE_RUN_SUITE_BASIC_TESTS` first then `NBL_NITE_RUN_SUITE_PERF_TESTS`. Once we pass the first we can say we have 100% working backend! (currently we get SegFault somewhere in half of `NBL_NITE_RUN_SUITE_BASIC_TESTS` because our backend is still WIP)

## GUI

If you want to pick individual tests and browse GUI just execute the target's executable with no arguments.

<video src="https://github.com/Devsh-Graphics-Programming/Nabla-Site-Media/raw/master/tools/nite/nite-gui.mp4"></video>
