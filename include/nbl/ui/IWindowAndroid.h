#ifndef _NBL_I_WINDOW_ANDROID_H_INCLUDED_
#define _NBL_I_WINDOW_ANDROID_H_INCLUDED_

#include "nbl/ui/IWindow.h"

#ifdef _NBL_PLATFORM_ANDROID_
namespace nbl::ui
{

class NBL_API2 IWindowAndroid : public IWindow
{
    public:
        using native_handle_t = struct ANativeWindow*;
        virtual const native_handle_t& getNativeHandle() const = 0;

    protected:
        virtual ~IWindowAndroid() = default;
        inline IWindowAndroid(SCreationParams&& params) : IWindow(std::move(params)) {}
};

}
#endif // _NBL_PLATFORM_ANDROID_

#endif
