#ifdef _NBL_PLATFORM_LINUX_
#include <nbl/ui/CWindowX11.h>
#include <nbl/ui/CWindowManagerX11.h>

#include <X11/extensions/XI.h>

#if !__has_include(<X11/extensions/XInput.h>)
#error "Please install libxi-dev"
#endif

#include <X11/extensions/XInput.h>

#include "nbl/system/DefaultFuncPtrLoader.h"

namespace nbl::ui
{

int CWindowX11::printXErrorCallback(Display *Display, XErrorEvent *event)
{
    char msg[256];
    char msg2[256];

    snprintf(msg, 256, "%d", event->request_code);
    x11.pXGetErrorDatabaseText(Display, "XRequest", msg, "unknown", msg2, 256);
    x11.pXGetErrorText(Display, event->error_code, msg, 256);
    // os::Printer::log("X Error", msg, ELL_WARNING);
    // os::Printer::log("From call ", msg2, ELL_WARNING);
    return 0;
}

CWindowManagerX11::CWindowManagerX11()
{
    m_windowThreadManager.m_windowsMapPtr = &m_windowsMap;
}

void CWindowX11::processEvent(XEvent event)
{
    IEventCallback* eventCallback = getEventCallback();

    std::cout << event.type << std::endl;

    switch(event.type)
    {
        case ResizeRequest: // window resize event
        {
            XResizeRequestEvent& e = (XResizeRequestEvent&)event;
            if (e.width != m_width || e.height != m_height)
            {
                if (eventCallback->onWindowResized(this, e.width, e.height))
                {
                    x11.pXResizeWindow(m_dpy, m_native, e.width, e.height);
                }
            }
            break;
        }

        case ConfigureNotify:  // window move event
        {
            XConfigureRequestEvent& e = (XConfigureRequestEvent&)event;
             if (e.x != m_x || e.y != m_y)
            {
                if (eventCallback->onWindowMoved(this, e.x, e.y))
                {
                    x11.pXMoveWindow(m_dpy, m_native, e.x, e.y);
                }
            }
            break;
        }

        case MapNotify: // window show event
        {
            if (eventCallback->onWindowShown(this)) 
            {
                x11.pXMapWindow(m_dpy, m_native);
            }

            break;
        }

        case UnmapNotify: // hide window event
        {
            if (eventCallback->onWindowHidden(this) || eventCallback->onWindowMinimized(this)) 
            {
                x11.pXUnmapWindow(m_dpy, m_native);
            }

            break;
        }

        case FocusIn:
        {
            eventCallback->onGainedKeyboardFocus(this);
            break;
        }
        case FocusOut:
        {
            eventCallback->onLostKeyboardFocus(this);
            break;
        }

        case KeyPress:
        {
            // TODO
            break;
        }

        case KeyRelease:
        {
            //TODO
            break;
        }

        default:
        {
            x11.pXFlush(m_dpy);
            break;
        }
    }
}

core::smart_refctd_ptr<IWindow> CWindowManagerX11::createWindow(IWindow::SCreationParams&& creationParams)
{
    m_managerDpy = x11.pXOpenDisplay((char *)0);
    m_windowThreadManager.m_dpy = m_managerDpy;

    int32_t x = creationParams.x;
    int32_t y = creationParams.y;
    uint32_t w = creationParams.width;
    uint32_t h = creationParams.height;
    CWindowX11::E_CREATE_FLAGS flags = creationParams.flags;
    const std::string_view& caption = creationParams.windowCaption;

    int screen = DefaultScreen(m_managerDpy);

    auto system = creationParams.system;

    auto depth = DefaultDepth(m_managerDpy, screen);
    auto visual = DefaultVisual(m_managerDpy,screen);

    XSetWindowAttributes attributes;
    
    attributes.override_redirect = 0;
    attributes.background_pixel = WhitePixel(m_managerDpy, screen);
    attributes.border_pixel = BlackPixel(m_managerDpy, screen);

    Window wnd = x11.pXCreateWindow(
            m_managerDpy,
            RootWindow(m_managerDpy, screen),
            0, 0,
            w, h,
            0,
            depth,
            InputOutput,
            visual,
            CWBorderPixel | CWEventMask,
            &attributes
    );

    x11.pXMapWindow(m_managerDpy, wnd);
    
    auto result = core::make_smart_refctd_ptr<CWindowX11>(std::move(creationParams), system, this, m_managerDpy, wnd);

    CWindowX11* cw = result.get();

    m_windowsMap.insert(wnd, cw);
    m_windowThreadManager.start();

    return result;
}

void CWindowManagerX11::destroyWindow(IWindow* wnd)
{
}

core::vector<XID> CWindowManagerX11::getConnectedMice() const
{
    core::vector<XID> result;
    int deviceCount;
    XDeviceInfo* devices = xinput.pXListInputDevices(m_managerDpy, &deviceCount);
    for(int i = 0; i < deviceCount; i++)
    {
        XDeviceInfo device = devices[i];
        bool has_motion = false, has_buttons = false;
        for(int j = 0; j < device.num_classes; j++)
        {
            if(device.inputclassinfo[j].c_class == ButtonClass) has_buttons = true;
            else if(device.inputclassinfo[j].c_class == ValuatorClass) has_motion = true;
        }
        if(has_motion && has_buttons) result.push_back(device.id);
    }
    return result;
}

core::vector<XID> CWindowManagerX11::getConnectedKeyboards() const
{
    core::vector<XID> result;
    int deviceCount;
    XDeviceInfo* devices = xinput.pXListInputDevices(m_managerDpy, &deviceCount);
    for(int i = 0; i < deviceCount; i++)
    {
        XDeviceInfo device = devices[i];
        bool has_keys = false;
        for(int j = 0; j < device.num_classes; j++)
        {
            if(device.inputclassinfo[j].c_class == KeyClass) has_keys = true;
        }
        if(has_keys) result.push_back(device.id);
    }
    return result;
}

CWindowX11::CWindowX11(IWindow::SCreationParams&& creationParams, core::smart_refctd_ptr<system::ISystem>& sys, CWindowManagerX11* manager, Display* dpy, native_handle_t win) : IWindowX11(std::move(creationParams)), m_manager(std::move(manager)), m_dpy(dpy), m_native(win)
{
}

CWindowX11::~CWindowX11()
{
    // x11.pXDestroyWindow(m_dpy, m_native);
}

void CWindowManagerX11::CThreadHandler::work(lock_t& lock)
{
    x11.pXNextEvent(m_dpy, &m_event);

    Window* nativeWindow = &m_event.xany.window;
    CWindowX11* currentWin = m_windowsMapPtr->find(*nativeWindow);

    currentWin->processEvent(m_event);
}

IClipboardManager* CWindowX11::getClipboardManager()
{
    return nullptr;
}

ICursorControl* CWindowX11::getCursorControl()
{
    return nullptr;
}

}

#endif