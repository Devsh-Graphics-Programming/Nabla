// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_CORE_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_CORE_COMPILE_CONFIG_H_INCLUDED__

//! Irrlicht SDK Version
#define NABLA_VERSION_MAJOR 0
#define NABLA_VERSION_MINOR 3
#define NABLA_VERSION_REVISION 0
#define NABLA_VERSION_INTEGER (NABLA_VERSION_MAJOR*100 + NABLA_VERSION_MINOR*10 + NABLA_VERSION_REVISION)
// This flag will be defined only in SVN, the official release code will have
// it undefined
//#define IRRLICHT_VERSION_SVN -alpha
#define NABLA_SDK_VERSION "0.3.0-beta2"

#include <stdio.h> // TODO: Although included elsewhere this is required at least for mingw

//#define _NBL_TEST_WAYLAND

// this actually includes file depending on build type (Debug/Release)
#include "BuildConfigOptions.h"

#if defined(_NBL_PLATFORM_LINUX_) || defined(_NBL_PLATFORM_ANDROID_)
#   define _NBL_POSIX_API_ // Android is not 100% POSIX, but it's close enough
#elif defined(_NBL_PLATFORM_WINDOWS_)
#   define _NBL_WINDOWS_API_
#endif

#ifdef _NBL_TARGET_ARCH_ARM_
#   define __NBL_COMPILE_WITH_ARM_SIMD_ // NEON
#else // target arch x86
#   define __NBL_COMPILE_WITH_SSE3
#   define __NBL_COMPILE_WITH_X86_SIMD_ // SSE 4.2 
#   include <immintrin.h>
#endif

#ifdef _MSC_VER
#	define _ENABLE_EXTENDED_ALIGNED_STORAGE
#endif

//! Uncomment the following line if you want to ignore the deprecated warnings
//#define IGNORE_DEPRECATED_WARNING

#ifdef _NBL_WINDOWS_API_

// To build Nabla as a static library, you must define _NBL_STATIC_LIB_ in both the
// Nabla build, *and* in the user application, before #including <nabla.h>
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
