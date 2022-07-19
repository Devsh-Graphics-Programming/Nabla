// TODO this has to go to system namespace

#include "nbl/video/CWaylandCaller.h"

#ifdef _NBL_BUILD_WITH_WAYLAND

namespace nbl {
namespace ui
{

#define _NBL_WAYLAND_IFACES_LIST \
    wl_seat_interface,\
    wl_surface_interface,\
    wl_shm_pool_interface,\
    wl_buffer_interface,\
    wl_registry_interface,\
    wl_shell_surface_interface,\
    wl_region_interface,\
    wl_pointer_interface,\
    wl_keyboard_interface,\
    wl_compositor_interface,\
    wl_output_interface,\
    wl_shell_interface,\
    wl_shm_interface,\
    wl_data_device_interface,\
    wl_data_source_interface,\
    wl_data_offer_interface,\
    wl_data_device_manager_interface

static bool wayland_ifaces_loaded = false;

#define _NBL_DEF_WAYLAND_IFACE(iface) const struct wl_interface* CWaylandInterfaces::wliface_##iface = nullptr;
NBL_FOREACH(_NBL_DEF_WAYLAND_IFACE, _NBL_WAYLAND_IFACES_LIST)
#undef _NBL_DEF_WAYLAND_IFACE

impl::WaylandFuncPtrLoader::WaylandFuncPtrLoader()
{
    libclient = dlopen("libwayland-client.so", RTLD_LAZY);
    libegl = dlopen("libwayland-egl.so", RTLD_LAZY);
    libcursor = dlopen("libwayland-cursor.so", RTLD_LAZY);

    if (wayland_ifaces_loaded)
        return;

#define _NBL_LOAD_WAYLAND_IFACE(iface) CWaylandInterfaces::wliface_##iface = reinterpret_cast<const struct wl_interface*>( this->loadFuncPtr(#iface) );
    NBL_FOREACH(_NBL_LOAD_WAYLAND_IFACE, _NBL_WAYLAND_IFACES_LIST)
#undef _NBL_LOAD_WAYLAND_IFACE

    wayland_ifaces_loaded = true;
}
#undef _NBL_WAYLAND_IFACES_LIST

}}

#endif