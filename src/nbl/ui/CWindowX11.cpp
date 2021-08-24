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

// CWindowX11::CWindowX11(core::smart_refctd_ptr<system::ISystem>&& sys, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags) : IWindowX11(std::move(sys), _w, _h, _flags), m_dpy(NULL), m_native(NULL)
// {

// }

void CWindowX11::processEvent(XEvent event)
{
    IEventCallback* eventCallback = getEventCallback();

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
            // if(e.atom == SDL_VideoData::_NET_WM_STATE)
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
}

CWindowManagerX11::CWindowManagerX11()
{
    m_dpy = x11.pXOpenDisplay(nullptr);
}

// core::smart_refctd_ptr<IWindow> CWindowManagerX11::createWindow(IWindow::SCreationParams&& creationParams)
// {
//     int32_t x = creationParams.x;
//     int32_t y = creationParams.y;
//     uint32_t w = creationParams.width;
//     uint32_t h = creationParams.height;
//     CWindowX11::E_CREATE_FLAGS flags = creationParams.flags;
//     const std::string_view& caption = creationParams.windowCaption;
//     CWindowX11::native_handle_t* wnd;
    
//     return core::make_smart_refctd_ptr<IWindow>(this, m_dpy, wnd);
// }

void CWindowManagerX11::destroyWindow(IWindow* wnd)
{

}

core::vector<XID> CWindowManagerX11::getConnectedMice() const
{
    core::vector<XID> result;
    int deviceCount;
    XDeviceInfo* devices = xinput.pXListInputDevices(m_dpy, &deviceCount);
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
    XDeviceInfo* devices = xinput.pXListInputDevices(m_dpy, &deviceCount);
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

// CWindowX11::CWindowX11(CWindowManagerX11* manager, Display* dpy, native_handle_t win) : m_manager(manager), m_dpy(dpy), m_native(win)
// {

// }

CWindowX11::~CWindowX11()
{
    m_manager->destroyWindow(this);
}

}

#endif