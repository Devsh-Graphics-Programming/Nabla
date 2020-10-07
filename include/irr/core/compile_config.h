// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_CORE_COMPILE_CONFIG_H_INCLUDED__

//! Irrlicht SDK Version
#define NABLA_VERSION_MAJOR 0
#define NABLA_VERSION_MINOR 3
#define NABLA_VERSION_REVISION 0
// This flag will be defined only in SVN, the official release code will have
// it undefined
//#define IRRLICHT_VERSION_SVN -alpha
#define NABLA_SDK_VERSION "0.3.0-beta2"

#define NEW_MESHES

#include <stdio.h> // TODO: Although included elsewhere this is required at least for mingw

//! Uncomment this line to compile with the SDL device
//#define _NBL_COMPILE_WITH_SDL_DEVICE_
#ifdef NO_NBL_COMPILE_WITH_SDL_DEVICE_
#undef _NBL_COMPILE_WITH_SDL_DEVICE_
#endif

// this actually includes file depending on build type (Debug/Release)
#include "BuildConfigOptions.h"

#define NEW_SHADERS 1

#ifdef _NBL_TARGET_ARCH_ARM_
#   define __NBL_COMPILE_WITH_ARM_SIMD_ // NEON
#else // target arch x86
#   define __NBL_COMPILE_WITH_SSE3
#   define __NBL_COMPILE_WITH_X86_SIMD_ // SSE 4.2 
#   include <immintrin.h>
#endif


#if defined(_NBL_SERVER_)
#   define NO_NBL_COMPILE_WITH_VULKAN_
#   define NO_NBL_COMPILE_WITH_OPENGL_
#endif

#ifdef NO_NBL_COMPILE_WITH_OPENGL_
#   undef _NBL_COMPILE_WITH_OPENGL_
#endif

// The Windows platform and API support SDL and WINDOW device
#if defined(_NBL_PLATFORM_WINDOWS_)
#   define _NBL_WINDOWS_API_
#   define _NBL_COMPILE_WITH_WINDOWS_DEVICE_
#   if defined(_MSC_VER) && (_MSC_VER < 1300)
#       error "Only Microsoft Visual Studio 7.0 and later are supported."
#   endif
#endif

#if defined(_NBL_PLATFORM_LINUX_)
#   define _NBL_POSIX_API_
#   define _NBL_COMPILE_WITH_X11_DEVICE_
#endif

#ifdef _NBL_SERVER_
#   define NO_NBL_LINUX_X11_RANDR_
#endif

#ifdef _MSC_VER
#	define _ENABLE_EXTENDED_ALIGNED_STORAGE
#endif

//! Uncomment the following line if you want to ignore the deprecated warnings
//#define IGNORE_DEPRECATED_WARNING

#ifdef _NBL_WINDOWS_API_

// To build Irrlicht as a static library, you must define _NBL_STATIC_LIB_ in both the
// Irrlicht build, *and* in the user application, before #including <irrlicht.h>
#ifndef _NBL_STATIC_LIB_
#ifdef NABLA_EXPORTS
#define NABLA_API __declspec(dllexport)
#else
#define NABLA_API __declspec(dllimport)
#endif // NABLA_EXPORT
#else
#define NABLA_API
#endif // _NBL_STATIC_LIB_

// Declare the calling convention.
#if defined(_STDCALL_SUPPORTED)
#define NBLCALLCONV __stdcall
#else
#define NBLCALLCONV __cdecl
#endif // STDCALL_SUPPORTED

#else // _NBL_WINDOWS_API_

// Force symbol export in shared libraries built with gcc.
#if (__GNUC__ >= 4) && !defined(_NBL_STATIC_LIB_) && defined(NABLA_EXPORTS)
#define NABLA_API __attribute__ ((visibility("default")))
#else
#define NABLA_API
#endif

#define NBLCALLCONV

#endif // _NBL_WINDOWS_API_

#ifndef _NBL_WINDOWS_API_
#   undef _NBL_WCHAR_FILESYSTEM
#endif

#endif
