// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

static const char* const copyright = "Irrlicht Engine (c) 2002-2011 Nikolaus Gebhardt";

#include "BuildConfigOptions.h"

#ifdef _NBL_WINDOWS_
#include <windows.h>
#if defined(_NBL_DEBUG) && !defined(__GNUWIN32__) && !defined(_WIN32_WCE)
#include <crtdbg.h>
#endif  // _NBL_DEBUG
#endif

#include "nabla.h"
#ifdef _NBL_COMPILE_WITH_WINDOWS_DEVICE_
#include "CIrrDeviceWin32.h"
#endif

#ifdef _NBL_COMPILE_WITH_X11_DEVICE_
#include "CIrrDeviceLinux.h"
#endif

#ifdef _NBL_COMPILE_WITH_SDL_DEVICE_
#include "CIrrDeviceSDL.h"
#endif

#include "CIrrDeviceConsole.h"

namespace nbl
{
//! stub for calling createDeviceEx
core::smart_refctd_ptr<IrrlichtDevice> createDevice(video::E_DRIVER_TYPE driverType,
    const core::dimension2d<uint32_t>& windowSize,
    uint32_t bits, bool fullscreen,
    bool stencilbuffer, bool vsync, IEventReceiver* res)
{
    SIrrlichtCreationParameters p;
    p.DriverType = driverType;
    p.WindowSize = windowSize;
    p.Bits = (uint8_t)bits;
    p.Fullscreen = fullscreen;
    p.Stencilbuffer = stencilbuffer;
    p.Vsync = vsync;
    p.EventReceiver = res;

    return createDeviceEx(p);
}

core::smart_refctd_ptr<IrrlichtDevice> createDeviceEx(const SIrrlichtCreationParameters& params)
{
    core::smart_refctd_ptr<IrrlichtDevice> dev;

#ifdef _NBL_COMPILE_WITH_WINDOWS_DEVICE_
    if(params.DeviceType == EIDT_WIN32 || (!dev && params.DeviceType == EIDT_BEST))
        dev = core::make_smart_refctd_ptr<CIrrDeviceWin32>(params);
#endif
#ifdef _NBL_COMPILE_WITH_X11_DEVICE_
    if(params.DeviceType == EIDT_X11 || (!dev && params.DeviceType == EIDT_BEST))
        dev = core::make_smart_refctd_ptr<CIrrDeviceLinux>(params);
#endif
#ifdef _NBL_COMPILE_WITH_SDL_DEVICE_
    if(params.DeviceType == EIDT_SDL || (!dev && params.DeviceType == EIDT_BEST))
        dev = core::make_smart_refctd_ptr<CIrrDeviceSDL>(params);
#endif
    if(params.DeviceType == EIDT_CONSOLE || (!dev && params.DeviceType == EIDT_BEST))
        dev = core::make_smart_refctd_ptr<CIrrDeviceConsole>(params);

    if(dev && !dev->getVideoDriver() && params.DriverType != video::EDT_NULL)
    {
        dev->closeDevice();  // destroy window
        dev->run();  // consume quit message
        dev = nullptr;
    }

    return dev;
}

}  // end namespace nbl

#if defined(_NBL_WINDOWS_API_)

BOOL APIENTRY DllMain(HANDLE hModule,
    DWORD ul_reason_for_call,
    LPVOID lpReserved)
{
    // _crtBreakAlloc = 139;

    switch(ul_reason_for_call)
    {
        case DLL_PROCESS_ATTACH:
#if defined(_NBL_DEBUG) && !defined(__GNUWIN32__) && !defined(__BORLANDC__)
            _CrtSetDbgFlag(_CRTDBG_LEAK_CHECK_DF | _CRTDBG_ALLOC_MEM_DF);
#endif
            break;
        case DLL_THREAD_ATTACH:
        case DLL_THREAD_DETACH:
        case DLL_PROCESS_DETACH:
            break;
    }
    return TRUE;
}

#endif  // defined(_NBL_WINDOWS_)
