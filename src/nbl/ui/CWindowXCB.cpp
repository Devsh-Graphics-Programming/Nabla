#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/string/StringLiteral.h"

#include "nbl/system/DefaultFuncPtrLoader.h"

#include "nbl/ui/IWindowXCB.h"
#include "nbl/ui/CWindowXCB.h"
#include "nbl/ui/CCursorControlXCB.h"
#include "nbl/ui/CClipboardManagerXCB.h"
#include "nbl/ui/CWindowManagerXCB.h"

#include <cstdint>
#include <string>
#include <array>
#include <string_view>
#include <variant>

namespace nbl::ui {

static bool checkXcbCookie(const Xcb& functionTable, xcb_connection_t* connection, xcb_void_cookie_t cookie) {
    if (xcb_generic_error_t* error = functionTable.pxcb_request_check(connection, cookie))
    {
        printf("XCB error: %d", error->error_code);
        return false;
    }
    return true;
}

CWindowXCB::CDispatchThread::CDispatchThread(CWindowXCB& window):
    m_window(window) {
}

void CWindowXCB::CDispatchThread::work(lock_t& lock){
    if(m_quit) {
        return; 
    }
    auto& xcb = m_window.m_windowManager->getXcbFunctionTable();
    auto& connection = m_window.m_connection;

    auto MW_DELETE_WINDOW = connection->resolveAtom(m_window.m_WM_DELETE_WINDOW);
    auto NET_WM_PING = connection->resolveAtom(m_window.m_NET_WM_PING);

    if(auto event = xcb.pxcb_wait_for_event(connection->getRawConnection())) {
        auto* eventCallback = m_window.getEventCallback();
        m_window.m_clipboardManager->process(&m_window, event);
        switch (event->response_type & ~0x80)
        {
            case 0: {
                xcb_generic_error_t* error = reinterpret_cast<xcb_generic_error_t*>(event);
                printf("XCB error: %d", error->error_code);
                break;
            }
            case XCB_CONFIGURE_NOTIFY: {
                xcb_configure_notify_event_t* cne = reinterpret_cast<xcb_configure_notify_event_t*>(event);
                if(m_window.m_width != cne->width || 
                    m_window.m_height != cne->height) {
                    eventCallback->onWindowResized(&m_window, cne->width, cne->height);
                }
                if(m_window.m_x != cne->x || 
                    m_window.m_y != cne->y) {
                    eventCallback->onWindowMoved(&m_window, cne->x, cne->y);
                }
                break;
            }
            case XCB_DESTROY_WINDOW: {
                xcb_destroy_window_request_t* dwr = reinterpret_cast<xcb_destroy_window_request_t*>(event);
                if(dwr->window == m_window.m_xcbWindow) {
                    m_quit = true;
                    eventCallback->onWindowClosed(&m_window);
                }
                break;
            }
            case XCB_CLIENT_MESSAGE: {
                xcb_client_message_event_t* cme = reinterpret_cast<xcb_client_message_event_t*>(event);
                if(cme->data.data32[0] == MW_DELETE_WINDOW) {
                    xcb.pxcb_unmap_window(m_window.getXcbConnection(), m_window.m_xcbWindow);
                    xcb.pxcb_destroy_window(m_window.getXcbConnection(), m_window.m_xcbWindow);
                    xcb.pxcb_flush(m_window.getXcbConnection());
                    m_window.m_xcbWindow = 0;
                    m_quit = true; // we need to quit the dispatch thread
                    eventCallback->onWindowClosed(&m_window);
                } else if(cme->data.data32[0] == NET_WM_PING && cme->window != connection->primaryScreen()->root) {
                    xcb_client_message_event_t ev = *cme;
                    ev.response_type = XCB_CLIENT_MESSAGE;
                    ev.window = m_window.m_xcbWindow;
                    ev.type = NET_WM_PING;
                    xcb.pxcb_send_event(m_window.getXcbConnection(), 0, m_window.m_xcbWindow, XCB_EVENT_MASK_NO_EVENT, reinterpret_cast<const char*>(&ev));
                    xcb.pxcb_flush(m_window.getXcbConnection());
                }
                break;
            }
        }
        free(event);
    }
}


void CWindowXCB::CDispatchThread::init()
{

}

void CWindowXCB::CDispatchThread::exit()
{
}

CWindowManagerXCB::CWindowManagerXCB() {
}

core::smart_refctd_ptr<IWindow> CWindowManagerXCB::createWindow(IWindow::SCreationParams&& creationParams)
{
    std::string title = std::string(creationParams.windowCaption);
    auto window = core::make_smart_refctd_ptr<CWindowXCB>(core::smart_refctd_ptr<CWindowManagerXCB>(this), std::move(creationParams));
    window->setCaption(title);
    return window;
}

bool CWindowManagerXCB::setWindowSize_impl(IWindow* window, uint32_t width, uint32_t height) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowSize_impl(width, height);
    return true;
}

bool CWindowManagerXCB::setWindowPosition_impl(IWindow* window, int32_t x, int32_t y) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowPosition_impl(x, y);
    return true;
}

bool CWindowManagerXCB::setWindowRotation_impl(IWindow* window, bool landscape) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowRotation_impl(landscape);
    return true;
}

bool CWindowManagerXCB::setWindowVisible_impl(IWindow* window, bool visible) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowVisible_impl(visible);
    return true;
}

bool CWindowManagerXCB::setWindowMaximized_impl(IWindow* window, bool maximized) {
    auto wnd = static_cast<IWindowXCB*>(window);
    wnd->setWindowMaximized_impl(maximized);
    return true;
}

CWindowXCB::CWindowXCB(core::smart_refctd_ptr<CWindowManagerXCB>&& winManager, SCreationParams&& params):
    IWindowXCB(std::move(params)),
    m_windowManager(winManager),
    m_connection(core::make_smart_refctd_ptr<XCBConnection>(core::smart_refctd_ptr<CWindowManagerXCB>(m_windowManager))),
    m_cursorControl(core::make_smart_refctd_ptr<CCursorControlXCB>(core::smart_refctd_ptr<XCBConnection>(m_connection))),
    m_clipboardManager(core::make_smart_refctd_ptr<CClipboardManagerXCB>(core::smart_refctd_ptr<XCBConnection>(m_connection))),
    m_dispatcher(*this) {
    
    auto& xcb = m_windowManager->getXcbFunctionTable();
    auto& xcbIccm = m_windowManager->getXcbIcccmFunctionTable();

    m_xcbWindow = xcb.pxcb_generate_id(m_connection->getRawConnection());

    const auto* primaryScreen = m_connection->primaryScreen();

    uint32_t eventMask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
    uint32_t valueList[] = {
        primaryScreen->black_pixel,
        XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_KEY_PRESS | XCB_EVENT_MASK_KEY_RELEASE |
            XCB_EVENT_MASK_FOCUS_CHANGE | XCB_EVENT_MASK_PROPERTY_CHANGE
    };

    xcb_void_cookie_t xcbCheckResult = xcb.pxcb_create_window(
        m_connection->getRawConnection(), XCB_COPY_FROM_PARENT, m_xcbWindow, primaryScreen->root,
        static_cast<int16_t>(m_x),
        static_cast<int16_t>(m_y),
        static_cast<int16_t>(m_width),
        static_cast<int16_t>(m_height), 4,
        XCB_WINDOW_CLASS_INPUT_OUTPUT, primaryScreen->root_visual, eventMask,
        valueList);

    setWindowSize_impl(m_width, m_height);
    
    auto WM_DELETE_WINDOW = m_connection->resolveAtom(m_WM_DELETE_WINDOW);
    auto NET_WM_PING = m_connection->resolveAtom(m_NET_WM_PING);
    auto WM_PROTOCOLS = m_connection->resolveAtom(m_WM_PROTOCOLS);
    
    const std::array atoms {WM_DELETE_WINDOW, NET_WM_PING};
    xcb.pxcb_change_property(
            m_connection->getRawConnection(), 
            XCB_PROP_MODE_REPLACE, 
            m_xcbWindow, 
            WM_PROTOCOLS, XCB_ATOM_ATOM, 32, atoms.size(), atoms.data());


    auto motifHints = fetchMotifMWHints(getFlags().value);
    m_connection->setMotifWmHints(m_xcbWindow, motifHints);
    
    if(isAlwaysOnTop()) {
        XCBConnection::XCBAtomToken<core::StringLiteral("NET_WM_STATE_ABOVE")> NET_WM_STATE_ABOVE;
        m_connection->setNetMWState(
            primaryScreen->root,
            m_xcbWindow, false, m_connection->resolveAtom(NET_WM_STATE_ABOVE));
    }

    xcb.pxcb_map_window(m_connection->getRawConnection(), m_xcbWindow);
    xcb.pxcb_flush(m_connection->getRawConnection());
    m_dispatcher.start();
}

CWindowXCB::~CWindowXCB()
{
}

IClipboardManager* CWindowXCB::getClipboardManager() {
    return m_clipboardManager.get();
}

ICursorControl* CWindowXCB::getCursorControl() {
    return m_cursorControl.get();
}

IWindowManager* CWindowXCB::getManager() {
    return m_windowManager.get();
}

bool CWindowXCB::setWindowSize_impl(uint32_t width, uint32_t height) {
    auto& xcb = m_windowManager->getXcbFunctionTable();
    auto& xcbIccm = m_windowManager->getXcbIcccmFunctionTable();

    xcb_size_hints_t hints = {0};

    xcbIccm.pxcb_icccm_size_hints_set_size(&hints, true, width, height);
    if(!isResizable()) {
        xcbIccm.pxcb_icccm_size_hints_set_min_size(&hints, width, height);
        xcbIccm.pxcb_icccm_size_hints_set_max_size(&hints, width, height);
    }
    xcbIccm.pxcb_icccm_set_wm_normal_hints(m_connection->getRawConnection(), m_xcbWindow, &hints);
    return  true;
}

bool CWindowXCB::setWindowPosition_impl(int32_t x, int32_t y) {
    auto& xcb = m_windowManager->getXcbFunctionTable();

    const int32_t values[] = { x, y };
    auto cookie = xcb.pxcb_configure_window_checked(m_connection->getRawConnection(), m_xcbWindow, XCB_CONFIG_WINDOW_X | XCB_CONFIG_WINDOW_Y, values);
    bool check = checkXcbCookie(xcb, m_connection->getRawConnection(), cookie);
    xcb.pxcb_flush(m_connection->getRawConnection());
    assert(check);
    return true;
}

void CWindowXCB::setCaption(const std::string_view& caption) {
    auto& xcb = m_windowManager->getXcbFunctionTable();

    xcb.pxcb_change_property(m_connection->getRawConnection(), XCB_PROP_MODE_REPLACE, m_xcbWindow, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, static_cast<uint32_t>(caption.size()), reinterpret_cast<const void* const>(caption.data()));
    xcb.pxcb_flush(m_connection->getRawConnection());
}

bool CWindowXCB::setWindowRotation_impl(bool landscape) {
    return true;
}

bool CWindowXCB::setWindowVisible_impl( bool visible) {
    auto& xcb = m_windowManager->getXcbFunctionTable();

    if(visible) {
        xcb.pxcb_map_window(m_connection->getRawConnection(), m_xcbWindow);
        xcb.pxcb_flush(m_connection->getRawConnection());
    } else {
        xcb.pxcb_unmap_window(m_connection->getRawConnection(), m_xcbWindow);
        xcb.pxcb_flush(m_connection->getRawConnection());
    }
    return true;
}

bool CWindowXCB::setWindowMaximized_impl(bool maximized) {
    auto& xcb = m_windowManager->getXcbFunctionTable();
    const auto* primaryScreen = m_connection->primaryScreen();

    m_connection->setNetMWState(
        primaryScreen->root,
            m_xcbWindow, maximized && !isBorderless(), m_connection->resolveAtom(m_NET_WM_STATE_FULLSCREEN));

    m_connection->setNetMWState(
        primaryScreen->root,
            m_xcbWindow, maximized && isBorderless(), 
            m_connection->resolveAtom(m_NET_WM_STATE_MAXIMIZED_VERT),
            m_connection->resolveAtom(m_NET_WM_STATE_MAXIMIZED_HORZ));

    xcb.pxcb_flush(m_connection->getRawConnection());
    return true;
}

}

#endif