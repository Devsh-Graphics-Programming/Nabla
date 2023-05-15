#ifndef _NBL_I_WINDOW_WIN32_H_INCLUDED_
#define _NBL_I_WINDOW_WIN32_H_INCLUDED_

#include "nbl/ui/IWindowManagerWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_
// forward declare HWND
struct HWND__;

namespace nbl::ui
{

class NBL_API2 IWindowWin32 : public IWindow
{
    public:
        using native_handle_t = HWND__*;
        virtual const native_handle_t& getNativeHandle() const = 0;
        
    protected:
        inline IWindowWin32(SCreationParams&& params) : IWindow(std::move(params)) {}
        virtual ~IWindowWin32() = default;
};

}

#endif

#endif
