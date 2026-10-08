#ifndef _NBL_I_WINDOW_XCB_H_INCLUDED_
#define _NBL_I_WINDOW_XCB_H_INCLUDED_

#include "nbl/ui/IWindowManagerXcb.h"

#ifdef _NBL_PLATFORM_LINUX_
// forward declare, so the public headers don't need <xcb/xcb.h>
struct xcb_connection_t;

namespace nbl::ui
{

class NBL_API2 IWindowXcb : public IWindow
{
    public:
        // An X11 window is only meaningful together with the connection it was created on
        struct native_handle_t
        {
            xcb_connection_t* connection = nullptr;
            uint32_t window = 0u; // `xcb_window_t`
        };
        virtual const native_handle_t& getNativeHandle() const = 0;

    protected:
        inline IWindowXcb(SCreationParams&& params) : IWindow(std::move(params)) {}
        virtual ~IWindowXcb() = default;
};

}

#endif

#endif
