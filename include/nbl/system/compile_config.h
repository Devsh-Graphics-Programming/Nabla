// Copyright (C) 2018-2020 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef __NBL_SYSTEM_COMPILE_CONFIG_H_INCLUDED__
#define __NBL_SYSTEM_COMPILE_CONFIG_H_INCLUDED__

#if defined(_NBL_PLATFORM_LINUX_)
#define _NBL_COMPILE_WITH_X11_DEVICE_
#endif

//! VidMode is ANCIENT
//#define NO_NBL_LINUX_X11_VIDMODE_

//! On some Linux systems the XF86 vidmode extension or X11 RandR are missing. Use these flags
//! to remove the dependencies such that Irrlicht will compile on those systems, too.
//! If you don't need colored cursors you can also disable the Xcursor extension
#if defined(_NBL_PLATFORM_LINUX_) && defined(_NBL_COMPILE_WITH_X11_)
#define _NBL_LINUX_X11_VIDMODE_
#define _NBL_LINUX_X11_RANDR_
#endif

//! Define _NBL_COMPILE_WITH_X11_ to compile the Irrlicht engine with X11 support.
/** If you do not wish the engine to be compiled with X11, comment this
define out. */
// Only used in LinuxDevice.
///#ifndef _NBL_SERVER_
#define _NBL_COMPILE_WITH_X11_
///#endif

#endif
