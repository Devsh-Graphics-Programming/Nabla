#include "nbl/ui/XCBConnection.h"

#ifdef _NBL_PLATFORM_LINUX_

namespace nbl::ui 
{
    XCBConnection::XCBConnection(core::smart_refctd_ptr<CWindowManagerXCB>&& windowManager):
        m_windowManager(std::move(windowManager)) {
            const auto& xcb = m_windowManager->getXcbFunctionTable();
            m_connection = xcb.pxcb_connect(nullptr, nullptr);
    }

    XCBConnection::~XCBConnection() {
        const auto&  xcb = m_windowManager->getXcbFunctionTable();
        xcb.pxcb_disconnect(m_connection);
    }

    void XCBConnection::setNetMWState(xcb_window_t rootWindow, xcb_window_t window, bool set, xcb_atom_t first, xcb_atom_t second) const {
        const auto& xcb = m_windowManager->getXcbFunctionTable();

        xcb_client_message_event_t event;
        event.response_type = XCB_CLIENT_MESSAGE;
        event.type = resolveAtom(m_NET_WM_STATE);
        event.window = window;
        event.format = 32;
        event.sequence = 0;
        event.data.data32[0] = set ? 1l : 0l;
        event.data.data32[1] = first;
        event.data.data32[2] = second;
        event.data.data32[3] = 1;
        event.data.data32[4] = 0;
        xcb.pxcb_send_event(m_connection, 0, rootWindow, 
            XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_SUBSTRUCTURE_REDIRECT, reinterpret_cast<const char*>(&event));
   
    }


    const xcb_screen_t* XCBConnection::primaryScreen() {
        const auto& xcb = m_windowManager->getXcbFunctionTable();
        const xcb_setup_t *setup = xcb.pxcb_get_setup(m_connection);
        xcb_screen_t *screen = xcb.pxcb_setup_roots_iterator(setup).data;
        return screen;
    }

    void XCBConnection::setMotifWmHints(xcb_window_t window, const MotifWmHints& hint) const {
        const auto& xcb = m_windowManager->getXcbFunctionTable();

        auto atomHint = resolveAtom(m_MOTIF_WM_HINTS);
        if(hint.flags != MotifFlags::MWM_HINTS_NONE) {
            xcb.pxcb_change_property(m_connection, XCB_PROP_MODE_REPLACE, window, 
                atomHint, 
                atomHint, 32, sizeof(MotifWmHints) / sizeof(uint32_t), &hint);
        } else {
            xcb.pxcb_delete_property(m_connection, window, atomHint);
        }
    }

}

#endif // _NBL_PLATFORM_LINUX_