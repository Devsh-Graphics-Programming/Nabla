// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_E_DEVICE_TYPES_H_INCLUDED__
#define __NBL_E_DEVICE_TYPES_H_INCLUDED__

namespace nbl
{
//! An enum for the different device types supported by the Irrlicht Engine.
enum E_DEVICE_TYPE
{

    //! A device native to Microsoft Windows
    /** This device uses the Win32 API and works in all versions of Windows. */
    EIDT_WIN32,

    //! A device native to Unix style operating systems.
    /** This device uses the X11 windowing system and works in Linux, Solaris, FreeBSD, OSX and
		other operating systems which support X11. */
    EIDT_X11,

    //! A device native to Mac OSX
    /** This device uses Apple's Cocoa API and works in Mac OSX 10.2 and above. */
    EIDT_OSX,

    //! A device for raw framebuffer access
    /** Best used with embedded devices and mobile systems.
		Does not need X11 or other graphical subsystems.
		May support hw-acceleration via OpenGL-ES for FBDirect */
    EIDT_FRAMEBUFFER,

    //! A simple text only device supported by all platforms.
    /** This device allows applications to run from the command line without opening a window.
		It only supports mouse and keyboard in Windows operating systems. */
    EIDT_CONSOLE,

    //! This selection allows Irrlicht to choose the best device from the ones available.
    /** If this selection is chosen then Irrlicht will try to use the IrrlichtDevice native
		to your operating system. If this is unavailable then the X11, SDL and then console device
		will be tried. This ensures that Irrlicht will run even if your platform is unsupported,
		although it may not be able to render anything. */
    EIDT_BEST
};

}  // end namespace nbl

#endif
