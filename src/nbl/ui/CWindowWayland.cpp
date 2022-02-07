#include "nbl/ui/CWindowWayland.h"

#ifdef _NBL_BUILD_WITH_WAYLAND

#include "nbl/ui/CWaylandCaller.h"

namespace nbl
{
namespace ui
{
static ui::CWaylandCaller wlcall;

const struct wl_registry_listener CWindowWayland::s_listener = {&CWindowWayland::registry_callback, nullptr};

void CWindowWayland::registry_callback(void* data, struct wl_registry* registry, uint32_t id, const char* interface, uint32_t version)
{
    using data_t = std::pair<struct wl_compositor*, struct wl_shell*>;

    data_t* d = reinterpret_cast<data_t*>(data);
    if(strcmp(interface, "wl_compositor") == 0)
    {
        d->first = reinterpret_cast<struct wl_compositor*>(wlcall.pwl_registry_bind(registry, id, ui::CWaylandInterfaces::wliface_wl_compositor_interface, 1));
    }
    else if(strcmp(interface, "wl_shell") == 0)
    {
        d->second = reinterpret_cast<struct wl_shell*>(wlcall.pwl_registry_bind(registry, id, ui::CWaylandInterfaces::wliface_wl_shell_interface, 1));
    }
}

CWindowWayland::CWindowWayland(core::smart_refctd_ptr<system::ISystem>&& sys, wl_display* dpy, native_handle_t win)
    : IWindowWayland(std::move(sys)), m_dpy(dpy), m_native(win)
{
    // TODO
    // get window extent
    // flags
}

CWindowWayland::CWindowWayland(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
    : IWindowWayland(std::move(sys), _w, _h, _flags)
{
    std::pair<struct wl_compositor*, struct wl_shell*> data;

    m_dpy = wlcall.pwl_display_connect(nullptr);
    assert(m_dpy != nullptr);

    struct wl_registry* registry = wlcall.pwl_display_get_registry(m_dpy);
    wlcall.pwl_registry_add_listener(registry, &s_listener, reinterpret_cast<void*>(&data));

    wlcall.pwl_display_dispatch(m_dpy);
    wlcall.pwl_display_roundtrip(m_dpy);
    struct wl_compositor* compositor = data.first;
    struct wl_shell* shell = data.second;
    assert(compositor);
    assert(shell);

    struct wl_surface* surface = wlcall.pwl_compositor_create_surface(compositor);
    assert(surface);

    struct wl_shell_surface* shellsurface = wlcall.pwl_shell_get_shell_surface(shell, surface);
    wlcall.pwl_shell_surface_set_toplevel(shellsurface);

    struct wl_egl_window* wnd = wlcall.pwl_egl_window_create(surface, _w, _h);

    m_native = wnd;
    // TODO flags
}
}
}

#endif
