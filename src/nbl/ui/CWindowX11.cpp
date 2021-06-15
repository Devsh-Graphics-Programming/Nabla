#include <CWindowX11.h>
#include <CWindowManagerX11.h>
//#ifdef _NBL_PLATFORM_LINUX_

#include <X11/extensions/XI.h>
#include <X11/extensions/XInput.h>

#include "nbl/system/DefaultFuncPtrLoader.h"

namespace nbl {
namespace ui
{

// TODO @sadiuk
// we will need those to be moved to system or ui namespace
// because X11 clipboard manager will need to use X11 calls
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
    ,XUniqueContext
    ,XSaveContext
);
#ifdef _NBL_LINUX_X11_RANDR_
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xrandr, system::DefaultFuncPtrLoader
    ,XF86VidModeModeInfo
    ,XF86VidModeSwitchToMode
    ,XF86VidModeSetViewPort
    ,XF86VidModeQueryExtension
    ,XF86VidModeGetAllModeLines
    ,XF86VidModeSwitchToMode
    ,XF86VidModeSetViewPort
);
#endif
#ifdef _NBL_LINUX_X11_VIDMODE_
NBL_SYSTEM_DECLARE_DYNAMIC_FUNCTION_CALLER_CLASS(Xxf86vm, system::DefaultFuncPtrLoader
    ,XRRGetScreenInfo
    ,XRRSetScreenConfig
    ,XRRFreeScreenConfigInfo
    ,XRRQueryExtension
    ,XRRGetScreenInfo
    ,XRRConfigSizes
    ,XRRSetScreenConfig
    ,XRRFreeScreenConfigInfo
);
#endif

namespace 
{
    X11 x11("X11");
#ifdef _NBL_LINUX_X11_RANDR_
    Xrandr xrandr("Xrandr");
#endif
#ifdef _NBL_LINUX_X11_VIDMODE_
    Xxf86vm xxf86vm("Xxf86vm");
#endif
}

int CWindowX11::printXErrorCallback(Display *Display, XErrorEvent *event)
{
    char msg[256];
    char msg2[256];

    snprintf(msg, 256, "%d", event->request_code);
    x11.pXGetErrorDatabaseText(Display, "XRequest", msg, "unknown", msg2, 256);
    x11.pXGetErrorText(Display, event->error_code, msg, 256);
    os::Printer::log("X Error", msg, ELL_WARNING);
    os::Printer::log("From call ", msg2, ELL_WARNING);
    return 0;
}

CWindowX11::CWindowX11(core::smart_refctd_ptr<system::ISystem>&& sys, Display* dpy, native_handle_t win) : IWindowX11(std::move(sys)), m_dpy(dpy), m_native(win)
{
    Window tmp;
    int x, y;
    unsigned int w, h, border, bits;

    x11.pXGetGeometry(m_dpy, win, &tmp, &x, &y, &w, &h, &border, &bits);

    m_width = w;
    m_height = h;

    m_x = x;
    m_y = y;

    // TODO m_flags
}

CWindowX11::CWindowX11(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags) : IWindowX11(std::move(sys), _w, _h, _flags), m_dpy(NULL), m_native(NULL)
{
    // XInitThreads() call not needed unless windows are created concurrently, spoof EGL synchronizes per-display access itself
    //"If all calls to Xlib functions are protected by some other access mechanism 
    //(for example, a mutual exclusion lock in a toolkit or through explicit client programming), Xlib thread initialization is not required."
    //XInitThreads();

    x11.pXSetErrorHandler(&printXErrorCallback);

    Display* dpy = x11.pXOpenDisplay(nullptr);
    int screennr = DefaultScreen(dpy);

    if (isFullscreen()) // contents of this if are most likely broken
    {
#ifdef _NBL_LINUX_X11_VIDMODE_
        xxf86m.pXF86VidModeModeInfo oldVidMode;
        xxf86m.pXF86VidModeSwitchToMode(dpy, screennr, &oldVidMode);
        xxf86m.pXF86VidModeSetViewPort(dpy, screennr, 0, 0);
#endif

#ifdef _NBL_LINUX_X11_RANDR_
        SizeID oldRandrMode;
        Rotation oldRandrRotation;
        XRRScreenConfiguration *config = xrandr.pXRRGetScreenInfo(dpy,DefaultRootWindow(dpy));
        xrandr.pXRRSetScreenConfig(dpy,config,DefaultRootWindow(dpy),oldRandrMode,oldRandrRotation,CurrentTime);
        xrandr.pXRRFreeScreenConfigInfo(config);
#endif

        int32_t eventbase, errorbase;
        int32_t bestMode = -1;

#ifdef _NBL_LINUX_X11_VIDMODE_
        if (xxf86m.pXF86VidModeQueryExtension(dpy, &eventbase, &errorbase))
        {
            int32_t modeCount;
            XF86VidModeModeInfo** modes;

            xxf86m.pXF86VidModeGetAllModeLines(dpy, screennr, &modeCount, &modes);

            // find fitting mode
            for (int32_t i = 0; i<modeCount; ++i)
            {
                if (bestMode==-1 && modes[i]->hdisplay >= m_width && modes[i]->vdisplay >= m_height)
                    bestMode = i;
                else if (bestMode!=-1 &&
                        modes[i]->hdisplay >= m_width &&
                        modes[i]->vdisplay >= m_height &&
                        modes[i]->hdisplay <= modes[bestMode]->hdisplay &&
                        modes[i]->vdisplay <= modes[bestMode]->vdisplay)
                    bestMode = i;
            }
            if (bestMode != -1)
            {
                xxf86m.pXF86VidModeSwitchToMode(dpy, screennr, modes[bestMode]);
                xxf86m.pXF86VidModeSetViewPort(dpy, screennr, 0, 0);
            }
            else
            {
                //os::Printer::log("Could not find specified video mode, running windowed.", ELL_WARNING);
                CreationParams.Fullscreen = false;
            }

            x11.pXFree(modes);
        }
        else
#endif
#ifdef _NBL_LINUX_X11_RANDR_
        if (xrandr.pXRRQueryExtension(dpy, &eventbase, &errorbase))
        {
            int32_t modeCount;
            XRRScreenConfiguration *config=xrandr.pXRRGetScreenInfo(dpy,DefaultRootWindow(dpy));
            XRRScreenSize *modes=xrandr.pXRRConfigSizes(config,&modeCount);
            for (int32_t i = 0; i<modeCount; ++i)
            {
                if (bestMode==-1 && (uint32_t)modes[i].width >= m_width && (uint32_t)modes[i].height >= m_height)
                    bestMode = i;
                else if (bestMode!=-1 &&
                        (uint32_t)modes[i].width >= m_width &&
                        (uint32_t)modes[i].height >= m_height &&
                        modes[i].width <= modes[bestMode].width &&
                        modes[i].height <= modes[bestMode].height)
                    bestMode = i;
            }
            if (bestMode != -1)
            {
                xrandr.pXRRSetScreenConfig(dpy,config,DefaultRootWindow(dpy),bestMode,oldRandrRotation,CurrentTime);
            }
            xrandr.pXRRFreeScreenConfigInfo(config);
        }
        else
#endif
        {
            m_flags = static_cast<E_CREATE_FLAGS>(m_flags & (~ECF_FULLSCREEN));
        }
    }

    XVisualInfo visTempl;
    int visNumber;

    visTempl.screen = screennr;
    visTempl.depth = 32;

    XVisualInfo* visual = x11.pXGetVisualInfo(dpy, VisualScreenMask | VisualDepthMask, &visTempl, &visNumber);

    Colormap colormap;
    colormap = x11.pXCreateColormap(dpy, RootWindow(dpy, visual->screen), visual->visual, AllocNone);

    XSetWindowAttributes attribs;
    attribs.colormap = colormap;
    attribs.border_pixel = 0;
    attribs.event_mask = StructureNotifyMask | FocusChangeMask | ExposureMask; // TODO depend on create flags?
    // if not ignore input:
    attribs.event_mask |= PointerMotionMask | ButtonPressMask | KeyPressMask | ButtonReleaseMask | KeyReleaseMask;
    attribs.override_redirect = isFullscreen();

    Window win;
    win = x11.pXCreateWindow(dpy, RootWindow(dpy, visual->screen), 0, 0, m_width, m_height, 0, visual->depth, InputOutput, visual->visual, 
            CWBorderPixel | CWColormap | CWEventMask | CWOverrideRedirect, &attribs);
    if (!isHidden())
        x11.pXMapRaised(dpy, win);
    Atom wmDelete;
    wmDelete = x11.pXInternAtom(dpy, "WM_DELETE_WINDOW", True);
    x11.pXSetWMProtocols(dpy, win, &wmDelete, 1);

    if (isFullscreen())
    {
        x11.pXSetInputFocus(dpy, win, RevertToParent, CurrentTime);
        int grabkb = x11.pXGrabKeyboard(dpy, win, True, GrabModeAsync, GrabModeAsync, CurrentTime);

        int grabptr = x11.pXGrabPointer(dpy, win, True, ButtonPressMask, GrabModeAsync, GrabModeAsync, win, None, CurrentTime);

        x11.pXWarpPointer(dpy, None, win, 0, 0, 0, 0, 0, 0);
    }

    m_dpy = dpy;
    m_native = win;

    XContext classId = x11.pXUniqueContext();
    x11.pXSaveContext(m_dpy, m_native, classId, (XPointer)this);

    int deviceCount;
    XDeviceInfo* infos =  XListInputDevices(m_dpy, &deviceCount);
}

void CWindowX11::processEvent(XEvent event)
{
    IEventCallback eventCallback = getEventCallback();

    switch(event.type)
    {
        case ConfigureNotify:
        {
            XConfigureEvent e = event.xconfigure;
            // Resized
            if(e.width != m_width || e.height != m_height)
            {
                eventCallback->onWindowResized(this, e.width, e.height);
                x11.pXResizeWindow(m_dpy, m_native, e.width, e.height);

            }
            // Moved
            if(e.x != m_x || e.y != m_y)
            {
                eventCallback->onWindowMoved(this, e.x, e.y);
                x11.pXMoveWindow(m_dpy, m_native, e.x, e.y);
            }

            break;
        }
        case MapNotify:
        {
            eventCallback->onWindowShown(this);
            break;
        }
        // Don't think these 2 are the same, will return to them later, but onWindowHidden is definitely right here
        case UnmapNotify:
        {
            eventCallback->onWindowHidden(this);
            eventCallback->onWindowMinimized(this); 
            break;
        }
        case PropertyNotify:
        {
            XPropertyEvent e = event.xproperty;
            if(e.atom == _NET_WM_STATE)
            {
                Atom* allStates;
                unsigned long itemCount, bytesAfter;
                unsigned char *properties = NULL;
                //Retrieving all states
                XGetWindowProperty(m_dpy, m_native, _NET_WM_STATE, 0, LONG_MAX, False, AnyPropertyType, allStates, &itemCount, &bytesAfter, &properties);
                bool maximizedVertically = false, maximizedHorizontally = false; 
                for(int i = 0; i < itemCount; i++)
                {
                    if(allStates[i] == _NET_WM_STATE_MAXIMIZED_HORZ) maximizedHorizontally = true;
                    else if(allStates[i] == _NET_WM_STATE_MAXIMIZED_VERT) maximizedVertically = true;
                }
                if(maximizedVertically && maximizedHorizontally && !isMaximized)
                {
                    isMaximized = true;
                    eventCallback->onWindowMaximized(this);
                }
            }

            break;
        }
        // TODO: don't know yet how those behave and whether i should
        // call mouse/keyboard/both focus change 
        case FocusIn:
        {
            
            break;
        }
        case FocusOut:
        {
            
            break;
        }
    }
}

void CWindowManagerX11::init()
{
    //TODO: init x11/xinput loaders here
    display = x11.pXOpenDisplay(nullptr);
}

void CWindowManagerX11::process_request(SRequest& req)
{
    switch (req.type)
    {
    case ERT_CREATE_WINDOW:
    {

        break;
    }
    case ERT_DESTROY_WINDOW:
    {

        break;
    }
    }
}


void CWindowManagerX11::background_work(lock_t& lock)
{
	XEvent event;
	x11.pXNextEvent(display, event);
	Window nativeWindow = event.xany.window;
	XPointer windowCharPtr = nullptr;
	x11.pXFindContext(display, nativeWindow, &windowCharPtr);
	CWindowX11* currentWindow = static_cast<CWindowX11*>(windowCharPtr);
	auto* eventCallback = nativeWindow->getEventCallback();
	currentWindow->processEvent(event);
}

}}

#endif