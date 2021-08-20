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
        core::vector<XID> getConnectedMice() const;
        core::vector<XID> getConnectedKeyboards() const;

	    Display* m_dpy;   
};

}
#endif
#endif