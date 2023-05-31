#ifndef __NBL_I_WINDOW_XCB_H_INCLUDED__
#define __NBL_I_WINDOW_XCB_H_INCLUDED__

#include "nbl/ui/XCBHandle.h"
#ifdef _NBL_PLATFORM_LINUX_

#include "nbl/core/util/bitflag.h"

#include "nbl/ui/IWindow.h"

#include <xcb/xproto.h>

namespace nbl::ui
{

class NBL_API2 IWindowXCB : public IWindow
{
    protected:
        virtual ~IWindowXCB() = default;
        inline IWindowXCB(SCreationParams&& params) : IWindow(std::move(params)) {}

    public:
        using IWindow::IWindow;

        struct native_handle_t {
            xcb_window_t m_window;
            core::smart_refctd_ptr<xcb::XCBHandle> m_connection;
        };

        virtual const native_handle_t* getNativeHandle() const = 0;
};

}

#endif
#endif