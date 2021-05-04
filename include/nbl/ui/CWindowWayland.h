#ifndef __C_WINDOW_WAYLAND_H_INCLUDED__
#define __C_WINDOW_WAYLAND_H_INCLUDED__

#include "nbl/ui/IWindowWayland.h"

#ifdef _NBL_BUILD_WITH_WAYLAND

#include <utility>
#include <cstring>

namespace nbl {
namespace ui
{

class CWindowWayland : public IWindowWayland
{
	static void registry_callback(void* data, struct wl_registry* registry, uint32_t id, const char* interface, uint32_t version);
    static const struct wl_registry_listener s_listener;

public:
	explicit CWindowWayland(core::smart_refctd_ptr<system::ISystem>&& sys, wl_display* dpy, native_handle_t win);

    struct wl_display* getDisplay() const override { return m_dpy; }
	native_handle_t getNativeHandle() const override { return m_native; }

	static core::smart_refctd_ptr<CWindowWayland> create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowWayland* win = new CWindowWayland(std::move(sys), _w, _h, _flags);
		return core::smart_refctd_ptr<CWindowWayland>(win, core::dont_grab);
	}

private:
    CWindowWayland(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);

    struct wl_display* m_dpy;
    native_handle_t m_native;
};

}
}
#endif

#endif
