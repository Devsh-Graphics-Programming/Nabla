# What IrrExtensions are

Stuff which is not 100% necessary for a GPGPU engine, or could have 3 or 4 possible implementations
none of which would be optimal for all circumstances such as shadows or deferred rendering.

## General rules for extensions are:
1) **Put a LICENSE.md file at the root of your extension directory**
2) Include a short README.md outlining external API, but not as documentation
3) Put all your source code in the irr::ext::YourExtension namespace
4) Keep all your files in `./ext/YourExtension/`
5) Include your own files (even in your own .c/cpp files) by specifying their paths relative to the irrlicht root (see rule 2). So instead of your extension's source files (and the users source files) asking for `#include "YourHeader.h"` they should ask for `#include "../../irr/ext/YourExtension/YourHeader.h"
6) Provide a CMakeLists.txt
7) Add a `IRR_BUILD_YOUR_EXTENION` CMake option that controls the building of any examples using Your Extension and that it is set to OFF by default. This is to make sure that your extension does not harm everybody's build-time and should Your Extension's dependencies fail to pull/configure it doesn't affect the rest of the library.
8) Make sure your extension's PUBLIC headers (if its a header only or a source inclusion only extension then all headers) are installed to `${CMAKE_INSTALL_PREFIX}/include/irr/ext/YourExtension/`
9) Put any dependencies of Your Extension as a submodule in `./3rdparty`, make sure the version agrees (no two versions of the same dependency) if other extensions or core-library use the same dependency (for example, zlib, libpng, freetype, etc.). Exceptionally if no actively maintained/mirrored git repository of the dependency exists, you can put extracted source code in `./3rdparty`.
10) Have your CMake configure script init and pull the submodules of dependencies
11) Under no circumstances shall you have a separate submodule/directory for different platform builds of the same dependency!
12) Dependencies MUST BUILD OUT-OF-SOURCE (can never pollute and show up as modified or untracked git files)
13) Dependencies must build with CMake (if there is no CMakeLists, then make one, see OpenSSL)
14) If extension is not a library but a collection of C++ headers and sources, then make sure it compiles only through inclusion with end-user projects.
15) If the extension requires higher capability hardware than IrrlichtBAW then provide static function to check whether the hardware supports it at runtime
16) Document your extension headers Doxygen-style


## Additional rules for a library extension (only if it has dependencies that are libraries):
1) `IRR_BUILD_YOUR_EXTENION=OFF` must disable the building of your extension library
2) Must link statically! (It can have a shared target as well, but static target is obligatory)
3) It must link its dependencies statically! (this is the only reason why I allow extensions to be libraries in the first place)
4) If compiling as a library, it must support out-of-source-build and a static library target.
5) Do not expose (via public headers) or install your dependencies! (Unless @devsh grants an exceptional waiver)

@manhnt9 provided a nice CMake script for setting up an ext library cleanly https://github.com/buildaworldnet/IrrlichtBAW/pull/291#issuecomment-494223131


## Rules for exposed dependencies (if @devsh grants a waiver):
1) This is only for big complicated framework integrations such as CEGUI, PhysX, Bullet, etc. where writing a non-dependency-API-exposing extension would entail a lot of work, pointless wrapper code, and hinder the usability of the underlying dependency.
2) Only the necessary dependency headers CAN install to `${CMAKE_INSTALL_PREFIX}/include/irr/ext/YOUR_EXT_NAME/3rdparty/YOUR_EXTS_3RDPARTY_DEPENDENCY_NAME` (never any path without an `${CMAKE_INSTALL_PREFIX}/include/irr` in the prefix)
3) Only if the dependency library cannot be statically linked (examples; commercial DLL binary-only lib, LGPL lib) then its shared libraries can be installed to `${CMAKE_INSTALL_PREFIX}/lib/irr/ext/YOUR_EXT_NAME/3rdparty/YOUR_EXTS_3RDPARTY_DEPENDENCY` (never any path without an `${CMAKE_INSTALL_PREFIX}/lib/irr` in the prefix)

Example installed dependency:
```
Sample Header
./install/win64-gcc/include/irr/ext/Bullet/3rdparty/include/bullet/Bullet-C-Api.h
Sample lib
./install/win64-gcc/lib/irr/ext/Bullet/3rdparty/libBulletCollision.so
```


# Current List of Extensions:
+ AutoExposure
+ Compute Shader 2D Box Blur
+ Full-screen Triangle for Fragment Shader Post Processing
+ Debug Line Draw using streaming buffer
+ CEGUI integration
