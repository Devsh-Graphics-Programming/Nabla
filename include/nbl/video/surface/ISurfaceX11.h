#ifndef __NBL_I_SURFACE_LINUX_H_INCLUDED__
#define __NBL_I_SURFACE_LINUX_H_INCLUDED__

#include "nbl/asset/compile_config.h"

#ifdef _NBL_PLATFORM_LINUX_

#include <X11/Xlib.h>

namespace nbl {
namespace video
{

class ISurfaceX11
{
public:
    struct SCreationParams
    {
        Display* dpy;
        Window window;
    };

protected:
    explicit ISurfaceX11(SCreationParams&& params) : m_params(std::move(params)) {}

    SCreationParams m_params;
};

}
}

#endif

#endif