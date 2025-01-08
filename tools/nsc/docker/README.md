# NSC Docker Godbolt

## Run NSC tool straight from build directory in compiler explorer docker container!

Currently only Windows platform with target *x86_64* architecture is supported. Tested with Hyper-V isolation mode.

### Requirements

- [***Docker Desktop***](https://www.docker.com/products/docker-desktop/)

### How To

Switch docker to windows containers, configure CMake with `NBL_ENABLE_DOCKER_INTGRATION` option (recommended Visual Studio generator) & build `run-compiler-explorer` target. After the build completes type `localhost` in your browser.

![Containers for Windows](https://user-images.githubusercontent.com/65064509/152947300-affca592-35a7-4e4c-a7fc-2055ce1ba528.png)

