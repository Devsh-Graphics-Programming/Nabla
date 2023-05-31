#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/decl/smart_refctd_ptr.h"
#include "nbl/core/string/StringLiteral.h"

#include "nbl/system/DefaultFuncPtrLoader.h"

#include "nbl/ui/IWindowXCB.h"
#include "nbl/ui/CWindowXCB.h"
#include "nbl/ui/CCursorControlXCB.h"
#include "nbl/ui/CWindowManagerXCB.h"

#include <cstdint>
#include <string>
#include <array>
#include <string_view>
#include <variant>

namespace nbl::ui {

static bool checkXcbCookie(const IWindowManagerXCB::Xcb& functionTable, xcb_connection_t* connection, xcb_void_cookie_t cookie) {
    if (xcb_generic_error_t* error = functionTable.pxcb_request_check(connection, cookie))
    {
        printf("XCB error: %d", error->error_code);
        return false;
    }
    return true;
}

void CWindowXCB::CDispatchThread::work(lock_t& lock){
    if(m_quit) {
        return; 
    }
    auto& xcb = m_window.m_windowManager->getXcbFunctionTable();
    // auto& connection = m_window.m_handle;
    auto& windowHandle = m_window.m_handle;


    if(auto event = xcb.pxcb_wait_for_event(*windowHandle.m_connection)) {
        auto* eventCallback = m_window.getEventCallback();
        // m_window.m_clipboardManager->process(&m_window, event);
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
                if(dwr->window == windowHandle.m_window) {
                    m_quit = true;
                    eventCallback->onWindowClosed(&m_window);
                }
                break;
            }
            case XCB_CLIENT_MESSAGE: {
                xcb_client_message_event_t* cme = reinterpret_cast<xcb_client_message_event_t*>(event);
                if(cme->data.data32[0] == windowHandle.m_connection->WM_DELETE_WINDOW) {
                    xcb.pxcb_unmap_window(*windowHandle.m_connection, windowHandle.m_window);
                    xcb.pxcb_destroy_window(*windowHandle.m_connection, windowHandle.m_window);
                    xcb.pxcb_flush(*windowHandle.m_connection);
                    windowHandle.m_window = 0;
                    m_quit = true; // we need to quit the dispatch thread
                    eventCallback->onWindowClosed(&m_window);
                } else if(cme->data.data32[0] == windowHandle.m_connection->_NET_WM_PING && cme->window != xcb::primaryScreen(*windowHandle.m_connection)->root) {
                    xcb_client_message_event_t ev = *cme;
                    ev.response_type = XCB_CLIENT_MESSAGE;
                    ev.window = m_window.m_handle.m_window;
                    ev.type = windowHandle.m_connection->_NET_WM_PING;
                    xcb.pxcb_send_event(*windowHandle.m_connection, 0, m_window.m_handle.m_window, XCB_EVENT_MASK_NO_EVENT, reinterpret_cast<const char*>(&ev));
                    xcb.pxcb_flush(*windowHandle.m_connection);
                }
                break;
            }
        }
        free(event);
    }
}


CWindowXCB::CWindowXCB(native_handle_t&& handle, core::smart_refctd_ptr<CWindowManagerXCB>&& winManager, SCreationParams&& params):
    IWindowXCB(std::move(params)),
    m_handle(std::move(handle)),
    m_windowManager(std::move(winManager)),
    m_dispatcher(*this) {
    
    auto& xcb = m_handle.m_connection->getXcbFunctionTable();
    auto& xcbIccm = m_handle.m_connection->getXcbIcccmFunctionTable();

    xcb.pxcb_map_window(*m_handle.m_connection, m_handle.m_window);
    xcb.pxcb_flush(*m_handle.m_connection);

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

void CWindowXCB::setCaption(const std::string_view& caption) {
    auto& xcb = m_handle.m_connection->getXcbFunctionTable();
    xcb.pxcb_change_property(*m_handle.m_connection, XCB_PROP_MODE_REPLACE, m_handle.m_window, XCB_ATOM_WM_NAME, XCB_ATOM_STRING, 8, static_cast<uint32_t>(caption.size()), reinterpret_cast<const void* const>(caption.data()));
    xcb.pxcb_flush(*m_handle.m_connection);
}
}

#endif