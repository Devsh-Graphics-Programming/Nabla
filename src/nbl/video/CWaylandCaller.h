// TODO this has to go to system namespace

#ifndef __NBL_C_WAYLAND_CALLER_H_INCLUDED__
#define __NBL_C_WAYLAND_CALLER_H_INCLUDED__

#include "nbl/macros.h"
#include "nbl/system/FuncPtrLoader.h"
#include "nbl/system/DynamicFunctionCaller.h"

#ifdef _NBL_BUILD_WITH_WAYLAND

#include <wayland-client.h>
#include <wayland-server.h>
#include <wayland-client-protocol.h>
#include <wayland-egl.h>  // Wayland EGL MUST be included before EGL headers

namespace nbl
{
namespace ui
{
namespace impl
{
class WaylandFuncPtrLoader final : system::FuncPtrLoader
{
protected:
    void* libclient;
    void* libegl;
    void* libcursor;

public:
    WaylandFuncPtrLoader();

    WaylandFuncPtrLoader(WaylandFuncPtrLoader&& other)
    {
        libclient = libegl = libcursor = NULL;
        operator=(std::move(other));
    }
    ~WaylandFuncPtrLoader()
    {
        if(libclient)
            dlclose(libclient);
        if(libegl)
            dlclose(libegl);
        if(libcursor)
            dlclose(libcursor);
    }

    inline WaylandFuncPtrLoader& operator=(WaylandFuncPtrLoader&& other)
    {
        std::swap(libclient, other.libclient);
        std::swap(libegl, other.libegl);
        std::swap(libcursor, other.libcursor);
        return *this;
    }

    inline bool isLibraryLoaded() override final
    {
        return libclient != NULL && libegl != NULL && libcursor != NULL;
    }

    inline void* loadFuncPtr(const char* funcname) override final
    {
        for(void* lib : {libclient, libegl, libcursor})
            if(void* f = dlsym(lib, funcname); f)
                return f;
        return nullptr;
    }
};
}

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(CWaylandCaller, impl::WaylandFuncPtrLoader, wl_display_connect, wl_compositor_create_region, wl_region_add, wl_surface_set_opaque_region, wl_egl_window_create, wl_shell_surface_pong, wl_egl_window_resize, wl_registry_bind, wl_display_get_registry, wl_registry_add_listener, wl_display_dispatch, wl_display_roundtrip, wl_egl_window_destroy, wl_shell_surface_destroy, wl_surface_destroy, wl_compositor_create_surface, wl_shell_get_shell_surface, wl_shell_surface_set_toplevel, wl_display_dispatch_pending, wl_display_disconnect);

struct CWaylandInterfaces
{
    CWaylandInterfaces() = delete;

#define _NBL_DECL_WAYLAND_IFACE(iface) static const struct wl_interface* wliface_##iface

    _NBL_DECL_WAYLAND_IFACE(wl_seat_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_surface_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_shm_pool_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_buffer_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_registry_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_shell_surface_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_region_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_pointer_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_keyboard_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_compositor_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_output_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_shell_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_shm_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_data_device_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_data_source_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_data_offer_interface);
    _NBL_DECL_WAYLAND_IFACE(wl_data_device_manager_interface);

#undef _NBL_DECL_WAYLAND_IFACE
};

}
}
#endif  //_NBL_BUILD_WITH_WAYLAND

#endif