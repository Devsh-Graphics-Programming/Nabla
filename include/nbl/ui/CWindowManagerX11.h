#ifdef _NBL_PLATFORM_LINUX_
#ifndef C_WINDOW_MANAGER_X11
#define C_WINDOW_MANAGER_X11

#include <nbl/ui/IWindowManager.h>
#include <X11/Xlib.h>
#include <nbl/ui/CWindowX11.h>
#include <X11/Xresource.h>
#include <X11/extensions/XInput.h>
#include <X11/extensions/xf86vmode.h>
#include <string>

namespace nbl::ui
{

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(X11, system::DefaultFuncPtrLoader
    ,XSetErrorHandler
    ,XOpenDisplay
    ,XFree
    ,XGetVisualInfo
    ,XCreateColormap
    ,XCreateWindow
    ,XMapRaised
    ,XInternAtom
    ,XSetWMProtocols
    ,XSetInputFocus
    ,XGrabKeyboard
    ,XGrabPointer
    ,XWarpPointer
    ,XGetErrorDatabaseText
    ,XGetErrorText
    ,XGetGeometry
    ,XFindContext
    ,XrmUniqueQuark
    ,XSaveContext
    ,XResizeWindow
    ,XMoveWindow
);

// TODO add more
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xinput, system::DefaultFuncPtrLoader
	,XListInputDevices
	,XOpenDevice
	,XCloseDevice
	,XSetDeviceMode
	,XSelectExtensionEvent
	,XGetDeviceMotionEvents
	,XFreeDeviceMotionEvents	
);

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xrandr, system::DefaultFuncPtrLoader
    ,XF86VidModeSwitchToMode
    ,XF86VidModeSetViewPort
    ,XF86VidModeQueryExtension
    ,XF86VidModeGetAllModeLines
);

NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xxf86vm, system::DefaultFuncPtrLoader
    ,XRRGetScreenInfo
    ,XRRSetScreenConfig
    ,XRRFreeScreenConfigInfo
    ,XRRQueryExtension
    ,XRRConfigSizes
);

static X11 x11("X11");
static Xinput xinput("Xinput");
static Xrandr xrandr("Xrandr");
static Xxf86vm xxf86vm("Xxf86vm");

class CWindowManagerX11 : public IWindowManager
{
    public:
        CWindowManagerX11();
        ~CWindowManagerX11() override = default;

        core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override;
        void destroyWindow(IWindow* wnd) override;
    private:
        class CThreadHandler final : public system::IThreadHandler<CThreadHandler>
        {
            using base_t = system::IThreadHandler<CThreadHandler>;
		    friend base_t;

            public:
                CThreadHandler(Display* dpy) : m_dpy(dpy)
                {
                    this->start();
                }

                void createWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowX11::E_CREATE_FLAGS _flags, CWindowX11::native_handle_t* wnd, const std::string_view& caption)
                {
                    // TODO
                }

                void destroyWindow(CWindowX11::native_handle_t window)
                {
                    // TODO
                }

            private:
                void work(lock_t& lock) {} // TODO
                void init();
                bool wakeupPredicate() const { return true; }
                bool continuePredicate() const { return true; }
                Display* m_dpy;
        };

        core::vector<XID> getConnectedMice() const;
        core::vector<XID> getConnectedKeyboards() const;

	    Display* m_dpy;   
};

}
#endif
#endif