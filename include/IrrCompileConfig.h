// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h

#ifndef __IRR_COMPILE_CONFIG_H_INCLUDED__
#define __IRR_COMPILE_CONFIG_H_INCLUDED__

//! Irrlicht SDK Version
#define IRRLICHTBAW_VERSION_MAJOR 0
#define IRRLICHTBAW_VERSION_MINOR 3
#define IRRLICHTBAW_VERSION_REVISION 0
// This flag will be defined only in SVN, the official release code will have
// it undefined
//#define IRRLICHT_VERSION_SVN -alpha
#define IRRLICHTBAW_SDK_VERSION "0.3.0-beta2"

#define NEW_MESHES

#include <stdio.h> // TODO: Although included elsewhere this is required at least for mingw

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

// this actually includes file depending on build type (Debug/Release)
#include "BuildConfigOptions.h"


#ifdef _IRR_TARGET_ARCH_ARM_
#   define __IRR_COMPILE_WITH_ARM_SIMD_ // NEON
#else // target arch x86
#   define __IRR_COMPILE_WITH_SSE3
#   define __IRR_COMPILE_WITH_X86_SIMD_ // SSE 4.2 
#   include <immintrin.h>
#endif


#if defined(_IRR_SERVER_)
#   define NO_IRR_COMPILE_WITH_VULKAN_
#   define NO_IRR_COMPILE_WITH_OPENGL_
#   define NO_IRR_COMPILE_WITH_BURNINGSVIDEO_
#endif

#ifdef NO_IRR_COMPILE_WITH_OPENGL_
#   undef _IRR_COMPILE_WITH_OPENGL_
#endif
#ifdef NO_IRR_COMPILE_WITH_BURNINGSVIDEO_
#   undef _IRR_COMPILE_WITH_BURNINGSVIDEO_
#endif

// The Windows platform and API support SDL and WINDOW device
#if defined(_IRR_PLATFORM_WINDOWS_)
#   define _IRR_WINDOWS_API_
#   define _IRR_COMPILE_WITH_WINDOWS_DEVICE_
#   if defined(_MSC_VER) && (_MSC_VER < 1300)
#       error "Only Microsoft Visual Studio 7.0 and later are supported."
#   endif
#endif

#if defined(_IRR_PLATFORM_LINUX_)
#   define _IRR_POSIX_API_
#   define _IRR_COMPILE_WITH_X11_DEVICE_
#endif

#ifdef _IRR_SERVER_
#   define NO_IRR_LINUX_X11_RANDR_
#endif

//! VidMode is ANCIENT
//#define NO_IRR_LINUX_X11_VIDMODE_

//! On some Linux systems the XF86 vidmode extension or X11 RandR are missing. Use these flags
//! to remove the dependencies such that Irrlicht will compile on those systems, too.
//! If you don't need colored cursors you can also disable the Xcursor extension
#if defined(_IRR_PLATFORM_LINUX_) && defined(_IRR_COMPILE_WITH_X11_)
#   define _IRR_LINUX_X11_VIDMODE_
#   define _IRR_LINUX_X11_RANDR_
#   ifdef NO_IRR_LINUX_X11_VIDMODE_
#       undef _IRR_LINUX_X11_VIDMODE_
#   endif
#   ifdef NO_IRR_LINUX_X11_RANDR_
#       undef _IRR_LINUX_X11_RANDR_
#   endif
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

//! Maximum number of output buffers and streams a Transform Feedback Object can have
#define _IRR_XFORM_FEEDBACK_MAX_BUFFERS_ 4
#define _IRR_XFORM_FEEDBACK_MAX_STREAMS_ 4

//! Define _IRR_COMPILE_WITH_X11_ to compile the Irrlicht engine with X11 support.
/** If you do not wish the engine to be compiled with X11, comment this
define out. */
// Only used in LinuxDevice.
///#ifndef _IRR_SERVER_
#define _IRR_COMPILE_WITH_X11_
///#endif
#ifdef NO_IRR_COMPILE_WITH_X11_
#   undef _IRR_COMPILE_WITH_X11_
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

//! Define _IRR_COMPILE_WITH_LIBPNG_ to enable compiling the engine using libpng.
/** This enables the engine to read png images. If you comment this out,
the engine will no longer read .png images. */
#define _IRR_COMPILE_WITH_LIBPNG_
#ifdef NO_IRR_COMPILE_WITH_LIBPNG_
#undef _IRR_COMPILE_WITH_LIBPNG_
#endif

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
#ifndef NEW_MESHES
//! Define _IRR_COMPILE_WITH_OBJ_WRITER_ if you want to write .obj files
#define _IRR_COMPILE_WITH_OBJ_WRITER_
#ifdef NO_IRR_COMPILE_WITH_OBJ_WRITER_
#undef _IRR_COMPILE_WITH_OBJ_WRITER_
#endif
#endif // NEW_MESHES

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

#ifndef _IRR_WINDOWS_API_
#   undef _IRR_WCHAR_FILESYSTEM
#endif


#ifdef _IRR_DEBUG
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

#define _IRR_BAW_FORMAT_VERSION 1

//! @see @ref CBlobsLoadingManager
#define _IRR_ADD_BLOB_SUPPORT(BlobClassName, EnumValue, Function, ...) \
case asset::Blob::EnumValue:\
	return asset::BlobClassName::Function(__VA_ARGS__);

//! Used inside CBlobsLoadingManager. Adds support of given blob types.
#define _IRR_SUPPORTED_BLOBS(Function, ...) \
_IRR_ADD_BLOB_SUPPORT(RawBufferBlobV1, EBT_RAW_DATA_BUFFER, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(TexturePathBlobV1, EBT_TEXTURE_PATH, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(MeshBlobV1, EBT_MESH, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(SkinnedMeshBlobV1, EBT_SKINNED_MESH, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(MeshBufferBlobV1, EBT_MESH_BUFFER, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(SkinnedMeshBufferBlobV1, EBT_SKINNED_MESH_BUFFER, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(MeshDataFormatDescBlobV1, EBT_DATA_FORMAT_DESC, Function, __VA_ARGS__)\
_IRR_ADD_BLOB_SUPPORT(FinalBoneHierarchyBlobV1, EBT_FINAL_BONE_HIERARCHY, Function, __VA_ARGS__)

#endif // __IRR_COMPILE_CONFIG_H_INCLUDED__
