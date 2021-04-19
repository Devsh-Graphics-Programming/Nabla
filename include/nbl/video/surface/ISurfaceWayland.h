#ifndef __NBL_I_SURFACE_WAYLAND_H_INCLUDED__
#define __NBL_I_SURFACE_WAYLAND_H_INCLUDED__

#include "nbl/core/compile_config.h"

#ifdef _NBL_BUILD_WITH_WAYLAND

#include <wayland-client.h>
#include <wayland-server.h>
#include <wayland-client-protocol.h>

namespace nbl {
namespace video
{

class ISurfaceWayland
{
public:
    struct SCreationParams
    {
        wl_display* dpy;
        wl_egl_window* window;
    };

protected:
    explicit ISurfaceWayland(SCreationParams&& params) : m_params(std::move(params)) { }

    SCreationParams m_params;
};

}
}

#endif //_NBL_BUILD_WITH_WAYLAND

#endif