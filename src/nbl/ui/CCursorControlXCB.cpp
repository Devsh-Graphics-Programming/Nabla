
#include "nbl/ui/CCursorControlXCB.h"

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/IWindowXCB.h"

namespace nbl::ui
{
    void CCursorControlXCB::setVisible(bool visible) {
        // TODO: implement
    }

    bool CCursorControlXCB::isVisible() const {
        return true;
    }

    void CCursorControlXCB::setPosition(SPosition pos) {
        // auto& xcb = m_connection->getXcbFunctionTable();
        // const auto* primaryScreen = m_connection->primaryScreen();
        // xcb.pxcb_warp_pointer(m_connection->getNativeHandle(), XCB_NONE, primaryScreen->root, 0, 0, 0, 0, pos.x, pos.y);
        // xcb.pxcb_flush(m_connection->getNativeHandle());
    }

    void CCursorControlXCB::setRelativePosition(IWindow* window, SRelativePosition position) {
        // auto* windowXcb = static_cast<IWindowXCB*>(window);
        // auto& xcb = m_connection->getXcbFunctionTable();
        // auto* handle = windowXcb->getNativeHandle();
        // assert(handle && handle->m_window != XCB_WINDOW_NONE);
        
        // xcb.pxcb_warp_pointer(m_connection->getNativeHandle(), XCB_NONE, handle->m_window, 0, 0, 0, 0, position.x, position.y);
        // xcb.pxcb_flush(m_connection->getNativeHandle());
    }

    CCursorControlXCB::SPosition CCursorControlXCB::getPosition() {
        // auto& xcb = m_connection->getXcbFunctionTable();
        // xcb_query_pointer_cookie_t token = xcb.pxcb_query_pointer(m_connection->getNativeHandle(), m_connection->primaryScreen()->root);
        // if(auto reply = xcb.pxcb_query_pointer_reply(m_connection->getNativeHandle(), token, nullptr)) {
        //     core::SRAIIBasedExiter exitReply([reply]() -> void { 
        //         free(reply);
        //     });
        //     return {reply->root_x, reply->root_y};
        // }
        return {0, 0};
    }

    CCursorControlXCB::SRelativePosition CCursorControlXCB::getRelativePosition(IWindow* window) {
        auto* xcbWidnow = static_cast<IWindowXCB*>(window);
        auto* windowHandle = xcbWidnow->getNativeHandle();

        // auto& xcb = m_connection->getXcbFunctionTable();
        // xcb_query_pointer_cookie_t token = xcb.pxcb_query_pointer(m_connection->getNativeHandle(), windowHandle->m_window);
        // if(auto reply = xcb.pxcb_query_pointer_reply(m_connection->getNativeHandle(), token, nullptr)) {
        //     core::SRAIIBasedExiter exitReply([reply]() -> void { 
        //         free(reply);
        //     });
        //     return {static_cast<float>(reply->win_x), static_cast<float>(reply->win_y)};
        // }
        return {0, 0};
    }
}
#endif
