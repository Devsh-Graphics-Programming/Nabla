#ifndef __C_WINDOW_X11_H_INCLUDED__
#define __C_WINDOW_X11_H_INCLUDED__

#ifdef _NBL_PLATFORM_LINUX_x1
#include "nbl/ui/IWindowX11.h"


#include <X11/Xutil.h>
#include <X11/extensions/xf86vmode.h>
#include <X11/extensions/Xrandr.h>
#include "nbl_os.h"

namespace nbl::ui
{

class CWindowManagerX11;

class NBL_API2 CWindowX11 final : public IWindowX11
{
	static int printXErrorCallback(Display *Display, XErrorEvent *event);

public:
	explicit CWindowX11(core::smart_refctd_ptr<system::ISystem>&& sys, Display* dpy, native_handle_t win);
	explicit CWindowX11(CWindowManagerX11* manager, Display* dpy, native_handle_t win);
    ~CWindowX11();
	Display* getDisplay() const override { return m_dpy; }
	const native_handle_t& getNativeHandle() const override { return m_native; }

	static core::smart_refctd_ptr<CWindowX11> create(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowX11* win = new CWindowX11(std::move(sys), _w, _h, _flags);
		return core::smart_refctd_ptr<CWindowX11>(win, core::dont_grab);
	}

	void processEvent(XEvent event);
private:
    CWindowX11(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);

    Display* m_dpy;
    native_handle_t m_native;
	CWindowManagerX11* m_manager;
private:
	std::map<XID, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannel;
	std::map<XID, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannel;
};

}

#endif

#endif
