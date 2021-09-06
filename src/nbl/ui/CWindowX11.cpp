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
/*
    switch(event.type)
    {
        case ConfigureNotify:
        {
            XConfigureEvent e = event.xconfigure;
            // Resized
            if(e.width != m_width || e.height != m_height)
            {
                if(eventCallback->onWindowResized(this, e.width, e.height))
                    x11.pXResizeWindow(m_dpy, m_native, e.width, e.height);
                    std::cout << "Window resized: " << e.width << " " << e.height << std::endl;

            }
            // Moved
            if(e.x != m_x || e.y != m_y)
            {
                if(eventCallback->onWindowMoved(this, e.x, e.y))
                    x11.pXMoveWindow(m_dpy, m_native, e.x, e.y);
            }

            break;
        }
        case MapNotify:
        {
            if (eventCallback->onWindowShown(this))
            {
                assert(false);
                // ShowWindow call or sth like that
            }

            break;
        }
        // Don't think these 2 are the same, will return to them later, but onWindowHidden is definitely right here
        case UnmapNotify: //TODO
        {
            assert(false);
            // eventCallback->onWindowHidden(this);
            // eventCallback->onWindowMinimized(this); 
            break;
        }
        
        case PropertyNotify:
        {
            // XPropertyEvent e = event.xproperty;
            // if(e.atom == _NET_WM_STATE)
            // {
                // Atom* allStates;
                // unsigned long itemCount, bytesAfter;
                // unsigned char *properties = NULL;
                // //Retrieving all states
                // XGetWindowProperty(m_dpy, m_native, _NET_WM_STATE, 0, LONG_MAX, False, AnyPropertyType, allStates, &itemCount, &bytesAfter, &properties);
                // bool maximizedVertically = false, maximizedHorizontally = false; 
                // for(int i = 0; i < itemCount; i++)
                // {
                //     if(allStates[i] == _NET_WM_STATE_MAXIMIZED_HORZ) maximizedHorizontally = true;
                //     else if(allStates[i] == _NET_WM_STATE_MAXIMIZED_VERT) maximizedVertically = true;
                // }
                // if(maximizedVertically && maximizedHorizontally && !isMaximized)
                // {
                //     isMaximized = true;
                //     if (eventCallback->onWindowMaximized(this)
                //     {
                //         assert(false);
                //         // MaximizeWindow or sth like that
                //     }
                // }
            // }

            break;
        }
        
        // TODO: don't know yet how those behave and whether i should
        // call mouse/keyboard/both focus change 
        case FocusIn:
        {
            assert(false);
            break;
        }
        case FocusOut:
        {
            assert(false);
            break;
        }
    }
    */
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