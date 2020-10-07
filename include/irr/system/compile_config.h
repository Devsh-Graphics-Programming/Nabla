// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_SYSTEM_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_SYSTEM_COMPILE_CONFIG_H_INCLUDED__

#include <stdio.h> // TODO: Although included elsewhere this is required at least for mingw

//! Uncomment this line to compile with the SDL device
//#define _NBL_COMPILE_WITH_SDL_DEVICE_
#ifdef NO_NBL_COMPILE_WITH_SDL_DEVICE_
#undef _NBL_COMPILE_WITH_SDL_DEVICE_
#endif

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

//! VidMode is ANCIENT
//#define NO_NBL_LINUX_X11_VIDMODE_

//! On some Linux systems the XF86 vidmode extension or X11 RandR are missing. Use these flags
//! to remove the dependencies such that Irrlicht will compile on those systems, too.
//! If you don't need colored cursors you can also disable the Xcursor extension
#if defined(_NBL_PLATFORM_LINUX_) && defined(_NBL_COMPILE_WITH_X11_)
#   define _NBL_LINUX_X11_VIDMODE_
#   define _NBL_LINUX_X11_RANDR_
#   ifdef NO_NBL_LINUX_X11_VIDMODE_
#       undef _NBL_LINUX_X11_VIDMODE_
#   endif
#   ifdef NO_NBL_LINUX_X11_RANDR_
#       undef _NBL_LINUX_X11_RANDR_
#   endif
#endif

//! Define _NBL_COMPILE_WITH_X11_ to compile the Irrlicht engine with X11 support.
/** If you do not wish the engine to be compiled with X11, comment this
define out. */
// Only used in LinuxDevice.
///#ifndef _NBL_SERVER_
#define _NBL_COMPILE_WITH_X11_
///#endif
#ifdef NO_NBL_COMPILE_WITH_X11_
#   undef _NBL_COMPILE_WITH_X11_
#endif

#endif
