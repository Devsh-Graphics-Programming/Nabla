#ifndef __NBL_I_WINDOW_WIN32_H_INCLUDED__
#define __NBL_I_WINDOW_WIN32_H_INCLUDED__

#include "nbl/ui/IWindow.h"

#ifdef _NBL_PLATFORM_WINDOWS_

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace nbl {
namespace ui
{

class IWindowWin32 : public IWindow
{
protected:
    virtual ~IWindowWin32() = default;
    IWindowWin32(SCreationParams&& params) : IWindow(std::move(params)) {}

public:
    using IWindow::IWindow;

    using native_handle_t = HWND;

    virtual native_handle_t getNativeHandle() const = 0;
};

}
}

#endif

#endif
