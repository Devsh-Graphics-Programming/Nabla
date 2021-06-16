#ifndef __C_WINDOW_X11_H_INCLUDED__
#define __C_WINDOW_X11_H_INCLUDED__

#include "nbl/ui/IWindowX11.h"

#ifdef _NBL_PLATFORM_LINUX_

#include <X11/Xutil.h>
#ifdef _NBL_LINUX_X11_VIDMODE_
#include <X11/extensions/xf86vmode.h>
#endif
#ifdef _NBL_LINUX_X11_RANDR_
#include <X11/extensions/Xrandr.h>
#endif
#include "nbl_os.h"

namespace nbl {
namespace ui
{

class CWindowX11 final : public IWindowX11
{
	static int printXErrorCallback(Display *Display, XErrorEvent *event);

public:
	explicit CWindowX11(Display* dpy, native_handle_t win);

    Display* getDisplay() const override { return m_dpy; }
	native_handle_t getNativeHandle() const override { return m_native; }

	static core::smart_refctd_ptr<CWindowX11> create(uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowX11* win = new CWindowX11(_w, _h, _flags);
		return core::smart_refctd_ptr<CWindowX11>(win, core::dont_grab);
	}

private:
    CWindowX11(uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);

    Display* m_dpy;
    native_handle_t m_native;
};

}
}

#endif

#endif
