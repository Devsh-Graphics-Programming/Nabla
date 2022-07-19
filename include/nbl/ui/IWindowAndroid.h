#ifndef _NBL_I_WINDOW_ANDROID_H_INCLUDED_
#define _NBL_I_WINDOW_ANDROID_H_INCLUDED_

#include "nbl/ui/IWindow.h"

#ifdef _NBL_PLATFORM_ANDROID_

#include <android/native_window.h>

namespace nbl::ui
{

class NBL_API2 IWindowAndroid : public IWindow
{
    protected:
        virtual ~IWindowAndroid() = default;
        inline IWindowAndroid(SCreationParams&& params) : IWindow(std::move(params)) {}

    public:
        using IWindow::IWindow;

        using native_handle_t = struct ANativeWindow*;

        virtual const native_handle_t& getNativeHandle() const = 0;
};

}

#endif // _NBL_PLATFORM_ANDROID_

#endif
