
#include "nbl/ui/XCBConnection.h"
#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/CCursorControlXCB.h"
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
        auto& xcb = m_xcbConnection->getXcbFunctionTable();
        const auto* primaryScreen = m_xcbConnection->primaryScreen();
        xcb.pxcb_warp_pointer(m_xcbConnection->getRawConnection(), XCB_NONE, primaryScreen->root, 0, 0, 0, 0, pos.x, pos.y);
        xcb.pxcb_flush(m_xcbConnection->getRawConnection());
    }
    void CCursorControlXCB::setRelativePosition(IWindow* window, SRelativePosition position) {
        auto* windowXcb = static_cast<IWindowXCB*>(window);
        auto& xcb = m_xcbConnection->getXcbFunctionTable();
        auto xcbWindow = windowXcb->getXcbWindow();

        xcb.pxcb_warp_pointer(m_xcbConnection->getRawConnection(), XCB_NONE, xcbWindow, 0, 0, 0, 0, position.x, position.y);
        xcb.pxcb_flush(m_xcbConnection->getRawConnection());
    }

    CCursorControlXCB::SPosition CCursorControlXCB::getPosition() {
        auto& xcb = m_xcbConnection->getXcbFunctionTable();
        xcb_query_pointer_cookie_t token = xcb.pxcb_query_pointer(m_xcbConnection->getRawConnection(), m_xcbConnection->primaryScreen()->root);
        if(auto reply = xcb.pxcb_query_pointer_reply(m_xcbConnection->getRawConnection(), token, nullptr)) {
            core::SRAIIBasedExiter exitReply([&reply]() -> void { 
                free(reply);
            });
            return {reply->root_x, reply->root_y};
        }
        return {0, 0};
    }

    CCursorControlXCB::SRelativePosition CCursorControlXCB::getRelativePosition(IWindow* window) {

        return {0, 0};
    }
}
#endif
