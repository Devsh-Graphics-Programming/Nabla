// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_S_EXPOSED_VIDEO_DATA_H_INCLUDED__
#define __NBL_S_EXPOSED_VIDEO_DATA_H_INCLUDED__

namespace nbl
{
namespace video
{
//! structure for holding data describing a driver and operating system specific data.
/** This data can be retrived by IVideoDriver::getExposedVideoData(). Use this with caution.
This only should be used to make it possible to extend the engine easily without
modification of its source. Note that this structure does not contain any valid data, if
you are using the null device.
*/
struct SExposedVideoData
{
    SExposedVideoData()
    {
        OpenGLWin32.HDc = 0;
        OpenGLWin32.HRc = 0;
        OpenGLWin32.HWnd = 0;
    }
    explicit SExposedVideoData(void* Window)
    {
        OpenGLWin32.HDc = 0;
        OpenGLWin32.HRc = 0;
        OpenGLWin32.HWnd = Window;
    }

    union
    {
        struct
        {
            //! Private GDI Device Context.
            /** Get if for example with: HDC h = reinterpret_cast<HDC>(exposedData.OpenGLWin32.HDc) */
            void* HDc;

            //! Permanent Rendering Context.
            /** Get if for example with: HGLRC h = reinterpret_cast<HGLRC>(exposedData.OpenGLWin32.HRc) */
            void* HRc;

            //! Window handle.
            /** Get with for example with: HWND h = reinterpret_cast<HWND>(exposedData.OpenGLWin32.HWnd) */
            void* HWnd;
        } OpenGLWin32;

        struct
        {
            // XWindow handles
            void* X11Display;
            void* X11Context;
            unsigned long X11Window;
        } OpenGLLinux;
    };
};

}  // end namespace video
}  // end namespace nbl

#endif
