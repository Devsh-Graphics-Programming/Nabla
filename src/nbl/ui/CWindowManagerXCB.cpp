#include "nbl/ui/CWindowManagerXCB.h"

#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/ui/CWindowManagerXCB.h"
#include "nbl/ui/CWindowXCB.h"

using namespace nbl;
using namespace nbl::ui;

core::smart_refctd_ptr<IWindowManagerXCB> IWindowManagerXCB::create()
{
    return core::make_smart_refctd_ptr<CWindowManagerXCB>();
}

CWindowManagerXCB::CWindowManagerXCB() {
}


core::smart_refctd_ptr<IWindow> CWindowManagerXCB::createWindow(IWindow::SCreationParams&& creationParams)
{
    IWindowXCB::native_handle_t windowHandle = {
        0,
        core::make_smart_refctd_ptr<xcb::XCBHandle>(core::smart_refctd_ptr<CWindowManagerXCB>(this))
    };
    const auto* primaryScreen = xcb::primaryScreen(*windowHandle.m_connection);
    windowHandle.m_window = m_xcb.pxcb_generate_id(*windowHandle.m_connection);

    uint32_t eventMask = XCB_CW_BACK_PIXEL | XCB_CW_EVENT_MASK;
    uint32_t valueList[] = {
        primaryScreen->black_pixel,
        XCB_EVENT_MASK_STRUCTURE_NOTIFY | XCB_EVENT_MASK_KEY_PRESS | XCB_EVENT_MASK_KEY_RELEASE |
            XCB_EVENT_MASK_FOCUS_CHANGE | XCB_EVENT_MASK_PROPERTY_CHANGE
    };

    xcb_void_cookie_t result = m_xcb.pxcb_create_window(
        *windowHandle.m_connection, XCB_COPY_FROM_PARENT, windowHandle.m_window, primaryScreen->root,
        static_cast<int16_t>(creationParams.x),
        static_cast<int16_t>(creationParams.y),
        static_cast<int16_t>(creationParams.width),
        static_cast<int16_t>(creationParams.height), 4,
        XCB_WINDOW_CLASS_INPUT_OUTPUT, primaryScreen->root_visual, eventMask,
        valueList);
    if(m_xcb.pxcb_request_check(*windowHandle.m_connection, result)) {
        m_xcb.pxcb_destroy_window(*windowHandle.m_connection, windowHandle.m_window);
        m_xcb.pxcb_flush(*windowHandle.m_connection);
        return nullptr;
    }

    const std::array atoms {windowHandle.m_connection->WM_DELETE_WINDOW, windowHandle.m_connection->_NET_WM_PING};
    m_xcb.pxcb_change_property(
            *windowHandle.m_connection, 
            XCB_PROP_MODE_REPLACE, 
            windowHandle.m_window, 
            windowHandle.m_connection->WM_PROTOCOLS, XCB_ATOM_ATOM, 32, atoms.size(), atoms.data());

    auto motifHints = xcb::createFlagsToMotifWmHints(creationParams.flags);
    xcb::setMotifWmHints(*windowHandle.m_connection, windowHandle.m_window, motifHints);

    if(creationParams.flags & IWindow::E_CREATE_FLAGS::ECF_ALWAYS_ON_TOP) {
        xcb::setNetMWState(*windowHandle.m_connection, 
            primaryScreen->root, 
            windowHandle.m_window, 
            windowHandle.m_window, 
            windowHandle.m_connection->NET_WM_STATE_ABOVE);
    }

    std::string title = std::string(creationParams.windowCaption);
    auto window = core::make_smart_refctd_ptr<CWindowXCB>(std::move(windowHandle), core::smart_refctd_ptr<CWindowManagerXCB>(this), std::move(creationParams));
    window->setCaption(title);
    return window;
}

bool CWindowManagerXCB::setWindowSize_impl(IWindow* window, uint32_t width, uint32_t height) {
    auto wnd = static_cast<IWindowXCB*>(window);
    auto* nativeHandle = wnd->getNativeHandle();
    
    xcb_size_hints_t hints = {0};
    m_xcbIcccm.pxcb_icccm_size_hints_set_size(&hints, true, width, height);
    m_xcbIcccm.pxcb_icccm_size_hints_set_min_size(&hints, width, height);
    m_xcbIcccm.pxcb_icccm_size_hints_set_max_size(&hints, width, height);
    m_xcbIcccm.pxcb_icccm_set_wm_normal_hints(*nativeHandle->m_connection, nativeHandle->m_window, &hints);
    return true;
}

bool CWindowManagerXCB::setWindowPosition_impl(IWindow* window, int32_t x, int32_t y) {
    auto wnd = static_cast<IWindowXCB*>(window);
    auto* nativeHandle = wnd->getNativeHandle();
    
    const int32_t values[] = { x, y };
    auto cookie = m_xcb.pxcb_configure_window_checked(*nativeHandle->m_connection, nativeHandle->m_window, XCB_CONFIG_WINDOW_X | XCB_CONFIG_WINDOW_Y, values);
    bool check = xcb::checkCookie(*nativeHandle->m_connection, cookie);
    m_xcb.pxcb_flush(*nativeHandle->m_connection);

    return check;
}

bool CWindowManagerXCB::setWindowRotation_impl(IWindow* window, bool landscape) {
    auto wnd = static_cast<IWindowXCB*>(window);
    auto* nativeHandle = wnd->getNativeHandle();
    

    return true;
}

bool CWindowManagerXCB::setWindowVisible_impl(IWindow* window, bool visible) {
    auto wnd = static_cast<IWindowXCB*>(window);
    auto* handle = wnd->getNativeHandle();

    if(visible) {
        m_xcb.pxcb_map_window(*handle->m_connection, handle->m_window);
        m_xcb.pxcb_flush(*handle->m_connection);
    } else {
        m_xcb.pxcb_unmap_window(*handle->m_connection, handle->m_window);
        m_xcb.pxcb_flush(*handle->m_connection);
    }

    return true;
}

bool CWindowManagerXCB::setWindowMaximized_impl(IWindow* window, bool maximized) {
    auto wnd = static_cast<IWindowXCB*>(window);
    auto* handle = wnd->getNativeHandle();
    const auto* primaryScreen = xcb::primaryScreen(*handle->m_connection);

    xcb::setNetMWState(
        *handle->m_connection,
        primaryScreen->root,
            handle->m_window, maximized && !wnd->isBorderless(), handle->m_connection->_NET_WM_STATE_FULLSCREEN);

    xcb::setNetMWState(
        *handle->m_connection,
        primaryScreen->root,
            handle->m_window, maximized && wnd->isBorderless(), 
            handle->m_connection->_NET_WM_STATE_MAXIMIZED_VERT,
            handle->m_connection->_NET_WM_STATE_MAXIMIZED_HORZ);

    m_xcb.pxcb_flush(*handle->m_connection);
    return true;
}

#endif