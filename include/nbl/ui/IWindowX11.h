#ifndef __NBL_I_WINDOW_X11_H_INCLUDED__
#define __NBL_I_WINDOW_X11_H_INCLUDED__

#include "nbl/ui/IWindow.h"

#ifdef _NBL_PLATFORM_LINUX_

#include <X11/Xlib.h>

namespace nbl::ui
{

class IWindowX11 : public IWindow
{
protected:
    virtual ~IWindowX11() = default;

public:
    using IWindow::IWindow;

    using native_handle_t = Window;

    virtual const native_handle_t& getNativeHandle() const = 0;
    virtual Display* getDisplay() const = 0;
};

}

#endif

#endif