#ifndef __NBL_I_WINDOW_LINUX_H_INCLUDED__
#define __NBL_I_WINDOW_LINUX_H_INCLUDED__

#include "nbl/system/IWindow.h"

#ifdef _NBL_PLATFORM_LINUX_

#include <X11/Xlib.h>

namespace nbl {
namespace system
{

class IWindowLinux : public IWindow
{
protected:
    virtual ~IWindowLinux() = default;

public:
    using IWindow::IWindow;

    using native_handle_t = Window;

    virtual native_handle_t getNativeHandle() const = 0;
    virtual Display* getDisplay() const = 0;
};

}
}

#endif

#endif