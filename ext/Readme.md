# What IrrExtensions are

Stuff which is not 100% necessary for a GPGPU engine, or could have 3 or 4 possible implementations
none of which would be optimal for all circumstances such as shadows or deferred rendering.

General rules for extensions are:
1) **Put a LICENSE.md file at the root of your extension directory**
2) Put it in the irr::ext::YourExtension namespace
3) Keep all your files in `./ext/YourExtension/`
4) Add a `IRR_BUILD_YOUR_EXTENION` CMake option that controls the building of any examples using Your Extension (and libYourExtension if its made as a library -- see rule 5) and that it is set to OFF by default. This is to make sure that your extension does not harm everybody's build-time and should Your Extension's dependencies fail to pull/configure it doesn't affect the rest of the library.
5) Make it compile only through inclusion with end-user projects, or **exceptionally** as a STATIC library with CMake.
6) Add a CMake script to make sure your extension's PUBLIC headers (if its a header only or a source inclusion only extension then all headers) are installed to `${CMAKE_INSTALL_PREFIX}/irr/ext/YourExtension/`
7) Include your own files (even in your own .c/cpp files) by specifying their
   paths relative to the irrlicht root (see rule 2). So instead of your extension's source files (and the users source files) asking for `#include "YourHeader.h"` they should ask for `#include "../../irr/ext/YourExtension/YourHeader.h"
8) If the extension requires higher capability hardware than IrrlichtBAW then
   provide static function to check whether the hardware supports it at runtime
9) Include a short README.md outlining external API, but not as documentation
10) Document your extension headers Doxygen-style
11) If compiling as a library, it must support out-of-source-build and a static library target.
12) Put any dependencies of Your Extension as a submodule or a source directory (if no maintained git repository of the dependency exists) in `./3rdparty`, make sure the version agrees if other extensions or core-library use the same dependency (for example, zlib, libpng, freetype, etc.).
13) Make sure your dependencies can be linked statically!
14) Do not expose or install your dependencies! (Unless @devsh grants an exceptional waiver)


Current List of Extensions:
+ AutoExposure
+ Compute Shader 2D Box Blur
+ Full-screen Triangle for Fragment Shader Post Processing
+ Debug Line Draw using streaming buffer
+ CEGUI integration
