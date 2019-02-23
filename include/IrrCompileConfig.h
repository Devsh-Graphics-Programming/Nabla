// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_COMPILE_CONFIG_H_INCLUDED__
#define __IRR_COMPILE_CONFIG_H_INCLUDED__

//! Irrlicht SDK Version
#define IRRLICHT_VERSION_MAJOR 1
#define IRRLICHT_VERSION_MINOR 8
#define IRRLICHT_VERSION_REVISION 3
// This flag will be defined only in SVN, the official release code will have
// it undefined
//#define IRRLICHT_VERSION_SVN -alpha
#define IRRLICHT_SDK_VERSION "1.8.3-baw"

#define NEW_MESHES

#define __IRR_COMPILE_WITH_X86_SIMD_

#ifdef __IRR_COMPILE_WITH_X86_SIMD_
#define __IRR_COMPILE_WITH_SSE3

#include <immintrin.h>

#ifdef __SSE3__
#define __IRR_COMPILE_WITH_SSE3
#endif

#ifdef __SSE4_1__
#define __IRR_COMPILE_WITH_SSE4_1
#endif

#ifdef __AVX__
#define __IRR_COMPILE_WITH_AVX
#endif



#ifdef __IRR_COMPILE_WITH_AVX
#define SIMD_ALIGNMENT 32
#else
#define SIMD_ALIGNMENT 16
#endif // __IRR_COMPILE_WITH_AVX

#endif

#include <stdio.h> // TODO: Although included elsewhere this is required at least for mingw

//! The defines for different operating system are:
//! _IRR_XBOX_PLATFORM_ for XBox
//! _IRR_WINDOWS_ for all irrlicht supported Windows versions
//! _IRR_WINDOWS_API_ for Windows or XBox
//! _IRR_LINUX_PLATFORM_ for Linux (it is defined here if no other os is defined)
//! _IRR_SOLARIS_PLATFORM_ for Solaris
//! _IRR_OSX_PLATFORM_ for Apple systems running OSX
//! _IRR_POSIX_API_ for Posix compatible systems
//! Note: PLATFORM defines the OS specific layer, API can group several platforms

//! DEVICE is the windowing system used, several PLATFORMs support more than one DEVICE
//! Irrlicht can be compiled with more than one device
//! _IRR_COMPILE_WITH_WINDOWS_DEVICE_ for Windows API based device
//! _IRR_COMPILE_WITH_WINDOWS_CE_DEVICE_ for Windows CE API based device
//! _IRR_COMPILE_WITH_OSX_DEVICE_ for Cocoa native windowing on OSX
//! _IRR_COMPILE_WITH_X11_DEVICE_ for Linux X11 based device
//! _IRR_COMPILE_WITH_SDL_DEVICE_ for platform independent SDL framework
//! _IRR_COMPILE_WITH_CONSOLE_DEVICE_ for no windowing system, used as a fallback

//! Passing defines to the compiler which have NO in front of the _IRR definename is an alternative
//! way which can be used to disable defines (instead of outcommenting them in this header).
//! So defines can be controlled from Makefiles or Projectfiles which allows building
//! different library versions without having to change the sources.
//! Example: NO_IRR_COMPILE_WITH_X11_ would disable X11


//! Uncomment this line to compile with the SDL device
//#define _IRR_COMPILE_WITH_SDL_DEVICE_
#ifdef NO_IRR_COMPILE_WITH_SDL_DEVICE_
#undef _IRR_COMPILE_WITH_SDL_DEVICE_
#endif

//! Comment this line to compile without the fallback console device.
#define _IRR_COMPILE_WITH_CONSOLE_DEVICE_
#ifdef NO_IRR_COMPILE_WITH_CONSOLE_DEVICE_
#undef _IRR_COMPILE_WITH_CONSOLE_DEVICE_
#endif

//! WIN32 for Windows32
//! WIN64 for Windows64
// The windows platform and API support SDL and WINDOW device
#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
#define _IRR_WINDOWS_
#define _IRR_WINDOWS_API_
#define _IRR_COMPILE_WITH_WINDOWS_DEVICE_
#define NOMINMAX
#endif

#if defined(_MSC_VER) && (_MSC_VER < 1300)
#  error "Only Microsoft Visual Studio 7.0 and later are supported."
#endif

// XBox only suppots the native Window stuff
#if defined(_XBOX)
	#undef _IRR_WINDOWS_
	#define _IRR_XBOX_PLATFORM_
	#define _IRR_WINDOWS_API_
	//#define _IRR_COMPILE_WITH_WINDOWS_DEVICE_
	#undef _IRR_COMPILE_WITH_WINDOWS_DEVICE_
	//#define _IRR_COMPILE_WITH_SDL_DEVICE_

	#include <xtl.h>
#endif

#if defined(__APPLE__) || defined(MACOSX)
#if !defined(MACOSX)
#define MACOSX // legacy support
#endif
#define _IRR_OSX_PLATFORM_
#define _IRR_COMPILE_WITH_OSX_DEVICE_
#endif

#if !defined(_IRR_WINDOWS_API_) && !defined(_IRR_OSX_PLATFORM_)
#ifndef _IRR_SOLARIS_PLATFORM_
#define _IRR_LINUX_PLATFORM_
#endif
#define _IRR_POSIX_API_
///#ifndef BAW_SERVER
#define _IRR_COMPILE_WITH_X11_DEVICE_
///#endif
#endif


//! Temporary until we can figure something out
#if __LSB_VERSION__ < 50
#define NO_IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#endif


//! Temporary until we can figure something out
#if __LSB_VERSION__ < 50
#define NO_IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#endif

//! Define _IRR_COMPILE_WITH_JOYSTICK_SUPPORT_ if you want joystick events.
#define _IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#ifdef NO_IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#undef _IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#endif


//! Maximum number of textures and input images we can feed to a shader
/** These limits will most likely be below your GPU hardware limits
**/
#define _IRR_MATERIAL_MAX_TEXTURES_ 8
#define _IRR_MATERIAL_MAX_IMAGES_ 8

//! Maximum of other shader input (output) slots
#define _IRR_MATERIAL_MAX_DYNAMIC_SHADER_STORAGE_OBJECTS_ 4
#define _IRR_MATERIAL_MAX_SHADER_STORAGE_OBJECTS_ (16-_IRR_MATERIAL_MAX_DYNAMIC_SHADER_STORAGE_OBJECTS_) //opengl has one set of slots for both
#define _IRR_MATERIAL_MAX_DYNAMIC_UNIFORM_BUFFER_OBJECTS_ 8
#define _IRR_MATERIAL_MAX_UNIFORM_BUFFER_OBJECTS_ (24-_IRR_MATERIAL_MAX_DYNAMIC_UNIFORM_BUFFER_OBJECTS_) //opengl has one set of slots for both

//! Maximum number of bits allowed in the VAO Attribute Divisor
#define _IRR_VAO_MAX_ATTRIB_DIVISOR_BITS 1

//! Maximum number of output buffers and streams a Transform Feedback Object can have
#define _IRR_XFORM_FEEDBACK_MAX_BUFFERS_ 4
#define _IRR_XFORM_FEEDBACK_MAX_STREAMS_ 4



//! Define _IRR_COMPILE_WITH_OPENGL_ to compile the Irrlicht engine with OpenGL.
/** If you do not wish the engine to be compiled with OpenGL, comment this
define out. */
#ifndef BAW_SERVER
#define _IRR_COMPILE_WITH_OPENGL_
///#define _IRR_COMPILE_WITH_OPENCL_
#endif

#ifdef NO_IRR_COMPILE_WITH_OPENGL_
#undef _IRR_COMPILE_WITH_OPENGL_
#endif

//! Define _IRR_COMPILE_WITH_BURNINGSVIDEO_ to compile the Irrlicht engine with Burning's video driver
/** If you do not need this software driver, you can comment this define out. */
#define _IRR_COMPILE_WITH_BURNINGSVIDEO_
#ifdef NO_IRR_COMPILE_WITH_BURNINGSVIDEO_
#undef _IRR_COMPILE_WITH_BURNINGSVIDEO_
#endif

//! Define _IRR_COMPILE_WITH_X11_ to compile the Irrlicht engine with X11 support.
/** If you do not wish the engine to be compiled with X11, comment this
define out. */
// Only used in LinuxDevice.
///#ifndef BAW_SERVER
#define _IRR_COMPILE_WITH_X11_
///#endif
#ifdef NO_IRR_COMPILE_WITH_X11_
#undef _IRR_COMPILE_WITH_X11_
#endif

//! Define _IRR_OPENGL_USE_EXTPOINTER_ if the OpenGL renderer should use OpenGL extensions via function pointers.
/** On some systems there is no support for the dynamic extension of OpenGL
	via function pointers such that this has to be undef'ed. */
#if !defined(_IRR_OSX_PLATFORM_) && !defined(_IRR_SOLARIS_PLATFORM_)
#define _IRR_OPENGL_USE_EXTPOINTER_
#endif

#ifdef BAW_SERVER
#define NO_IRR_LINUX_X11_RANDR_
#endif
//! VidMode is ANCIENT
//#define NO_IRR_LINUX_X11_VIDMODE_
//! On some Linux systems the XF86 vidmode extension or X11 RandR are missing. Use these flags
//! to remove the dependencies such that Irrlicht will compile on those systems, too.
//! If you don't need colored cursors you can also disable the Xcursor extension
#if defined(_IRR_LINUX_PLATFORM_) && defined(_IRR_COMPILE_WITH_X11_)
#define _IRR_LINUX_X11_VIDMODE_
#define _IRR_LINUX_X11_RANDR_
#ifdef NO_IRR_LINUX_X11_VIDMODE_
#undef _IRR_LINUX_X11_VIDMODE_
#endif
#ifdef NO_IRR_LINUX_X11_RANDR_
#undef _IRR_LINUX_X11_RANDR_
#endif

//! X11 has by default only monochrome cursors, but using the Xcursor library we can also get color cursor support.
//! If you have the need for custom color cursors on X11 then enable this and make sure you also link
//! to the Xcursor library in your Makefile/Projectfile.
//#define _IRR_LINUX_XCURSOR_
#ifdef NO_IRR_LINUX_XCURSOR_
#undef _IRR_LINUX_XCURSOR_
#endif

#endif


//! Define _IRR_COMPILE_WITH_OPENSSL_ to enable compiling the engine using libjpeg.
/** This enables the engine to read and write encrypted BAW format files.
If you comment this out, the engine will no longer read or write encrypted .baw files! */
#define _IRR_COMPILE_WITH_OPENSSL_
#ifdef NO_IRR_COMPILE_WITH_OPENSSL_
#undef _IRR_COMPILE_WITH_OPENSSL_
#endif

//! Define _IRR_COMPILE_WITH_JPEGLIB_ to enable compiling the engine using libjpeg.
/** This enables the engine to read jpeg images. If you comment this out,
the engine will no longer read .jpeg images. */
#define _IRR_COMPILE_WITH_LIBJPEG_
#ifdef NO_IRR_COMPILE_WITH_LIBJPEG_
#undef _IRR_COMPILE_WITH_LIBJPEG_
#endif

//! Define _IRR_USE_NON_SYSTEM_JPEG_LIB_ to let irrlicht use the jpeglib which comes with irrlicht.
/** If this is commented out, Irrlicht will try to compile using the jpeg lib installed in the system.
	This is only used when _IRR_COMPILE_WITH_LIBJPEG_ is defined. */
#define _IRR_USE_NON_SYSTEM_JPEG_LIB_
#ifdef NO_IRR_USE_NON_SYSTEM_JPEG_LIB_
#undef _IRR_USE_NON_SYSTEM_JPEG_LIB_
#endif

//! Define _IRR_COMPILE_WITH_LIBPNG_ to enable compiling the engine using libpng.
/** This enables the engine to read png images. If you comment this out,
the engine will no longer read .png images. */
#define _IRR_COMPILE_WITH_LIBPNG_
#ifdef NO_IRR_COMPILE_WITH_LIBPNG_
#undef _IRR_COMPILE_WITH_LIBPNG_
#endif

//! Define _IRR_USE_NON_SYSTEM_LIBPNG_ to let irrlicht use the libpng which comes with irrlicht.
/** If this is commented out, Irrlicht will try to compile using the libpng installed in the system.
	This is only used when _IRR_COMPILE_WITH_LIBPNG_ is defined. */
#define _IRR_USE_NON_SYSTEM_LIB_PNG_
#ifdef NO_IRR_USE_NON_SYSTEM_LIB_PNG_
#undef _IRR_USE_NON_SYSTEM_LIB_PNG_
#endif


//! Define _IRR_USE_NVIDIA_PERFHUD_ to opt-in to using the nVidia PerHUD tool
/** Enable, by opting-in, to use the nVidia PerfHUD performance analysis driver
tool <http://developer.nvidia.com/object/nvperfhud_home.html>. */
#undef _IRR_USE_NVIDIA_PERFHUD_

//! Define one of the three setting for Burning's Video Software Rasterizer
/** So if we were marketing guys we could say Irrlicht has 4 Software-Rasterizers.
	In a Nutshell:
		All Burnings Rasterizers use 32 Bit Backbuffer, 32Bit Texture & 32 Bit Z or WBuffer,
		16 Bit/32 Bit can be adjusted on a global flag.

		BURNINGVIDEO_RENDERER_BEAUTIFUL
			32 Bit + Vertexcolor + Lighting + Per Pixel Perspective Correct + SubPixel/SubTexel Correct +
			Bilinear Texturefiltering + WBuffer

		BURNINGVIDEO_RENDERER_FAST
			32 Bit + Per Pixel Perspective Correct + SubPixel/SubTexel Correct + WBuffer +
			Bilinear Dithering TextureFiltering + WBuffer

		BURNINGVIDEO_RENDERER_ULTRA_FAST
			16Bit + SubPixel/SubTexel Correct + ZBuffer
*/

#define BURNINGVIDEO_RENDERER_BEAUTIFUL
//#define BURNINGVIDEO_RENDERER_FAST
//#define BURNINGVIDEO_RENDERER_ULTRA_FAST

//! Uncomment the following line if you want to ignore the deprecated warnings
//#define IGNORE_DEPRECATED_WARNING


//! Define _IRR_COMPILE_WITH_X_LOADER_ if you want to use Microsoft X files
#define _IRR_COMPILE_WITH_X_LOADER_
#ifdef NO_IRR_COMPILE_WITH_X_LOADER_
#undef _IRR_COMPILE_WITH_X_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_LMTS_LOADER_ if you want to load LMTools files
//#define _IRR_COMPILE_WITH_LMTS_LOADER_
#ifdef NO_IRR_COMPILE_WITH_LMTS_LOADER_
#undef _IRR_COMPILE_WITH_LMTS_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_MY3D_LOADER_ if you want to load MY3D files
//#define _IRR_COMPILE_WITH_MY3D_LOADER_
#ifdef NO_IRR_COMPILE_WITH_MY3D_LOADER_
#undef _IRR_COMPILE_WITH_MY3D_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_OBJ_LOADER_ if you want to load Wavefront OBJ files
#define _IRR_COMPILE_WITH_OBJ_LOADER_
#ifdef NO_IRR_COMPILE_WITH_OBJ_LOADER_
#undef _IRR_COMPILE_WITH_OBJ_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_OCT_LOADER_ if you want to load FSRad OCT files
//#define _IRR_COMPILE_WITH_OCT_LOADER_
#ifdef NO_IRR_COMPILE_WITH_OCT_LOADER_
#undef _IRR_COMPILE_WITH_OCT_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_LWO_LOADER_ if you want to load Lightwave3D files
//#define _IRR_COMPILE_WITH_LWO_LOADER_
#ifdef NO_IRR_COMPILE_WITH_LWO_LOADER_
#undef _IRR_COMPILE_WITH_LWO_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_STL_LOADER_ if you want to load stereolithography files
#define _IRR_COMPILE_WITH_STL_LOADER_
#ifdef NO_IRR_COMPILE_WITH_STL_LOADER_
#undef _IRR_COMPILE_WITH_STL_LOADER_
#endif
#define _IRR_COMPILE_WITH_BAW_LOADER_
#ifdef NO_IRR_COMPILE_WITH_BAW_LOADER_
#undef _IRR_COMPILE_WITH_BAW_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_PLY_LOADER_ if you want to load Polygon (Stanford Triangle) files
#define _IRR_COMPILE_WITH_PLY_LOADER_
#ifdef NO_IRR_COMPILE_WITH_PLY_LOADER_
#undef _IRR_COMPILE_WITH_PLY_LOADER_
#endif

//! Define _IRR_COMPILE_WITH_STL_WRITER_ if you want to write .stl files
#define _IRR_COMPILE_WITH_STL_WRITER_
#ifdef NO_IRR_COMPILE_WITH_STL_WRITER_
#undef _IRR_COMPILE_WITH_STL_WRITER_
#endif
//! Define _IRR_COMPILE_WITH_BAW_WRITER_ if you want to write .baw files
#define _IRR_COMPILE_WITH_BAW_WRITER_
#ifdef NO_IRR_COMPILE_WITH_BAW_WRITER_
#undef _IRR_COMPILE_WITH_BAW_WRITER_
#endif
//! Define _IRR_COMPILE_WITH_PLY_WRITER_ if you want to write .ply files
#define _IRR_COMPILE_WITH_PLY_WRITER_
#ifdef NO_IRR_COMPILE_WITH_PLY_WRITER_
#undef _IRR_COMPILE_WITH_PLY_WRITER_
#endif

//! Define _IRR_COMPILE_WITH_BMP_LOADER_ if you want to load .bmp files
//! Disabling this loader will also disable the built-in font
#define _IRR_COMPILE_WITH_BMP_LOADER_
#ifdef NO_IRR_COMPILE_WITH_BMP_LOADER_
#undef _IRR_COMPILE_WITH_BMP_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_JPG_LOADER_ if you want to load .jpg files
#define _IRR_COMPILE_WITH_JPG_LOADER_
#ifdef NO_IRR_COMPILE_WITH_JPG_LOADER_
#undef _IRR_COMPILE_WITH_JPG_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_PNG_LOADER_ if you want to load .png files
#define _IRR_COMPILE_WITH_PNG_LOADER_
#ifdef NO_IRR_COMPILE_WITH_PNG_LOADER_
#undef _IRR_COMPILE_WITH_PNG_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_DDS_LOADER_ if you want to load .dds files
// Outcommented because
// a) it doesn't compile on 64-bit currently
// b) anyone enabling it should be aware that S3TC compression algorithm which might be used in that loader
// is patented in the US by S3 and they do collect license fees when it's used in applications.
// So if you are unfortunate enough to develop applications for US market and their broken patent system be careful.
#define _IRR_COMPILE_WITH_DDS_LOADER_
#ifdef NO_IRR_COMPILE_WITH_DDS_LOADER_
#undef _IRR_COMPILE_WITH_DDS_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_TGA_LOADER_ if you want to load .tga files
#define _IRR_COMPILE_WITH_TGA_LOADER_
#ifdef NO_IRR_COMPILE_WITH_TGA_LOADER_
#undef _IRR_COMPILE_WITH_TGA_LOADER_
#endif
//! Define _IRR_COMPILE_WITH_RGB_LOADER_ if you want to load Silicon Graphics .rgb/.rgba/.sgi/.int/.inta/.bw files
#define _IRR_COMPILE_WITH_RGB_LOADER_
#ifdef NO_IRR_COMPILE_WITH_RGB_LOADER_
#undef _IRR_COMPILE_WITH_RGB_LOADER_
#endif

//! Define _IRR_COMPILE_WITH_BMP_WRITER_ if you want to write .bmp files
#define _IRR_COMPILE_WITH_BMP_WRITER_
#ifdef NO_IRR_COMPILE_WITH_BMP_WRITER_
#undef _IRR_COMPILE_WITH_BMP_WRITER_
#endif
//! Define _IRR_COMPILE_WITH_JPG_WRITER_ if you want to write .jpg files
#define _IRR_COMPILE_WITH_JPG_WRITER_
#ifdef NO_IRR_COMPILE_WITH_JPG_WRITER_
#undef _IRR_COMPILE_WITH_JPG_WRITER_
#endif
//! Define _IRR_COMPILE_WITH_PNG_WRITER_ if you want to write .png files
#define _IRR_COMPILE_WITH_PNG_WRITER_
#ifdef NO_IRR_COMPILE_WITH_PNG_WRITER_
#undef _IRR_COMPILE_WITH_PNG_WRITER_
#endif
//! Define _IRR_COMPILE_WITH_TGA_WRITER_ if you want to write .tga files
#define _IRR_COMPILE_WITH_TGA_WRITER_
#ifdef NO_IRR_COMPILE_WITH_TGA_WRITER_
#undef _IRR_COMPILE_WITH_TGA_WRITER_
#endif

//! Define __IRR_COMPILE_WITH_ZIP_ARCHIVE_LOADER_ if you want to open ZIP and GZIP archives
/** ZIP reading has several more options below to configure. */
#define __IRR_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
#ifdef NO__IRR_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
#undef __IRR_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
#endif
#ifdef __IRR_COMPILE_WITH_ZIP_ARCHIVE_LOADER_
//! Define _IRR_COMPILE_WITH_ZLIB_ to enable compiling the engine using zlib.
/** This enables the engine to read from compressed .zip archives. If you
disable this feature, the engine can still read archives, but only uncompressed
ones. */
#define _IRR_COMPILE_WITH_ZLIB_
#ifdef NO_IRR_COMPILE_WITH_ZLIB_
#undef _IRR_COMPILE_WITH_ZLIB_
#endif
//! Define _IRR_USE_NON_SYSTEM_ZLIB_ to let irrlicht use the zlib which comes with irrlicht.
/** If this is commented out, Irrlicht will try to compile using the zlib
installed on the system. This is only used when _IRR_COMPILE_WITH_ZLIB_ is
defined. */
#define _IRR_USE_NON_SYSTEM_ZLIB_
#ifdef NO_IRR_USE_NON_SYSTEM_ZLIB_
#undef _IRR_USE_NON_SYSTEM_ZLIB_
#endif
//! Define _IRR_COMPILE_WITH_ZIP_ENCRYPTION_ if you want to read AES-encrypted ZIP archives
#define _IRR_COMPILE_WITH_ZIP_ENCRYPTION_
#ifdef NO_IRR_COMPILE_WITH_ZIP_ENCRYPTION_
#undef _IRR_COMPILE_WITH_ZIP_ENCRYPTION_
#endif
//! Define _IRR_COMPILE_WITH_BZIP2_ if you want to support bzip2 compressed zip archives
/** bzip2 is superior to the original zip file compression modes, but requires
a certain amount of memory for decompression and adds several files to the
library. */
#define _IRR_COMPILE_WITH_BZIP2_
#ifdef NO_IRR_COMPILE_WITH_BZIP2_
#undef _IRR_COMPILE_WITH_BZIP2_
#endif
//! Define _IRR_USE_NON_SYSTEM_BZLIB_ to let irrlicht use the bzlib which comes with irrlicht.
/** If this is commented out, Irrlicht will try to compile using the bzlib
installed on the system. This is only used when _IRR_COMPILE_WITH_BZLIB_ is
defined. */
#define _IRR_USE_NON_SYSTEM_BZLIB_
#ifdef NO_IRR_USE_NON_SYSTEM_BZLIB_
#undef _IRR_USE_NON_SYSTEM_BZLIB_
#endif
//! Define _IRR_COMPILE_WITH_LZMA_ if you want to use LZMA compressed zip files.
/** LZMA is a very efficient compression code, known from 7zip. Irrlicht
currently only supports zip archives, though. */
//#define _IRR_COMPILE_WITH_LZMA_
#ifdef NO_IRR_COMPILE_WITH_LZMA_
#undef _IRR_COMPILE_WITH_LZMA_
#endif
#endif

//! Define __IRR_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_ if you want to mount folders as archives
#define __IRR_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_
#ifdef NO__IRR_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_
#undef __IRR_COMPILE_WITH_MOUNT_ARCHIVE_LOADER_
#endif
//! Define __IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_ if you want to open ID software PAK archives
#define __IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_
#ifdef NO__IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_
#undef __IRR_COMPILE_WITH_PAK_ARCHIVE_LOADER_
#endif
//! Define __IRR_COMPILE_WITH_NPK_ARCHIVE_LOADER_ if you want to open Nebula Device NPK archives
#define __IRR_COMPILE_WITH_NPK_ARCHIVE_LOADER_
#ifdef NO__IRR_COMPILE_WITH_NPK_ARCHIVE_LOADER_
#undef __IRR_COMPILE_WITH_NPK_ARCHIVE_LOADER_
#endif
//! Define __IRR_COMPILE_WITH_TAR_ARCHIVE_LOADER_ if you want to open TAR archives
#define __IRR_COMPILE_WITH_TAR_ARCHIVE_LOADER_
#ifdef NO__IRR_COMPILE_WITH_TAR_ARCHIVE_LOADER_
#undef __IRR_COMPILE_WITH_TAR_ARCHIVE_LOADER_
#endif
//! Define __IRR_COMPILE_WITH_WAD_ARCHIVE_LOADER_ if you want to open WAD archives
#define __IRR_COMPILE_WITH_WAD_ARCHIVE_LOADER_
#ifdef NO__IRR_COMPILE_WITH_WAD_ARCHIVE_LOADER_
#undef __IRR_COMPILE_WITH_WAD_ARCHIVE_LOADER_
#endif

//! Set FPU settings
/** Irrlicht should use approximate float and integer fpu techniques
precision will be lower but speed higher. currently X86 only
*/
#if !defined(_IRR_OSX_PLATFORM_) && !defined(_IRR_SOLARIS_PLATFORM_)
//	#define IRRLICHT_FAST_MATH
	#ifdef NO_IRRLICHT_FAST_MATH
	#undef IRRLICHT_FAST_MATH
	#endif
#endif

// Some cleanup and standard stuff

#ifdef _IRR_WINDOWS_API_

// To build Irrlicht as a static library, you must define _IRR_STATIC_LIB_ in both the
// Irrlicht build, *and* in the user application, before #including <irrlicht.h>
#ifndef _IRR_STATIC_LIB_
#ifdef IRRLICHT_EXPORTS
#define IRRLICHT_API __declspec(dllexport)
#else
#define IRRLICHT_API __declspec(dllimport)
#endif // IRRLICHT_EXPORT
#else
#define IRRLICHT_API
#endif // _IRR_STATIC_LIB_

// Declare the calling convention.
#if defined(_STDCALL_SUPPORTED)
#define IRRCALLCONV __stdcall
#else
#define IRRCALLCONV __cdecl
#endif // STDCALL_SUPPORTED

#else // _IRR_WINDOWS_API_

// Force symbol export in shared libraries built with gcc.
#if (__GNUC__ >= 4) && !defined(_IRR_STATIC_LIB_) && defined(IRRLICHT_EXPORTS)
#define IRRLICHT_API __attribute__ ((visibility("default")))
#else
#define IRRLICHT_API
#endif

#define IRRCALLCONV

#endif // _IRR_WINDOWS_API_

// We need to disable DIRECT3D9 support for Visual Studio 6.0 because
// those $%&$!! disabled support for it since Dec. 2004 and users are complaining
// about linker errors. Comment this out only if you are knowing what you are
// doing. (Which means you have an old DX9 SDK and VisualStudio6).
#ifdef _MSC_VER
#if (_MSC_VER < 1300 && !defined(__GNUC__))
#undef _IRR_COMPILE_WITH_DIRECT3D_9_
#pragma message("Compiling Irrlicht with Visual Studio 6.0, support for DX9 is disabled.")
#endif
#endif

// XBox does not have OpenGL or DirectX9
#if defined(_IRR_XBOX_PLATFORM_)
	#undef _IRR_COMPILE_WITH_OPENGL_
	#undef _IRR_COMPILE_WITH_DIRECT3D_9_
#endif


#ifndef _IRR_WINDOWS_API_
	#undef _IRR_WCHAR_FILESYSTEM
#endif


#if defined(_IRR_SOLARIS_PLATFORM_)
	#undef _IRR_COMPILE_WITH_JOYSTICK_EVENTS_
#endif

#ifdef _DEBUG
	//! A few attributes are written in CSceneManager when _IRR_SCENEMANAGER_DEBUG is enabled
	// NOTE: Those attributes were used always until 1.8.0 and became a global define for 1.8.1
	// which is only enabled in debug because it had a large (sometimes >5%) impact on speed.
	// A better solution in the long run is to break the interface and remove _all_ attribute
	// access in functions like CSceneManager::drawAll and instead put that information in some
	// own struct/class or in CSceneManager.
	// See http://irrlicht.sourceforge.net/forum/viewtopic.php?f=2&t=48211 for the discussion.
	//#define _IRR_SCENEMANAGER_DEBUG
	#ifdef NO_IRR_SCENEMANAGER_DEBUG
		#undef _IRR_SCENEMANAGER_DEBUG
	#endif
#endif

//! @see @ref CBlobsLoadingManager
#define _IRR_ADD_BLOB_SUPPORT(BlobClassName, EnumValue, Function, ...) \
case core::Blob::EnumValue:\
	return core::BlobClassName::Function(__VA_ARGS__);

//! Used inside CBlobsLoadingManager. Adds support of given blob types.
#define _IRR_SUPPORTED_BLOBS(Function, ...) \
_IRR_ADD_BLOB_SUPPORT(RawBufferBlobV0, EBT_RAW_DATA_BUFFER, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(TexturePathBlobV0, EBT_TEXTURE_PATH, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(MeshBlobV0, EBT_MESH, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(SkinnedMeshBlobV0, EBT_SKINNED_MESH, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(MeshBufferBlobV0, EBT_MESH_BUFFER, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(SkinnedMeshBufferBlobV0, EBT_SKINNED_MESH_BUFFER, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(MeshDataFormatDescBlobV0, EBT_DATA_FORMAT_DESC, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(FinalBoneHierarchyBlobV0, EBT_FINAL_BONE_HIERARCHY, Function, __VA_ARGS__)

#endif // __IRR_COMPILE_CONFIG_H_INCLUDED__

