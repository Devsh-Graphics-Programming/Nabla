#ifndef __NBL_C_WINDOW_ANDROID_H_INCLUDED__
#define __NBL_C_WINDOW_ANDROID_H_INCLUDED__

#include "nbl/ui/IWindowAndroid.h"

#ifdef _NBL_PLATFORM_ANDROID_

namespace nbl {
namespace ui
{

class CWindowAndroid : public IWindowAndroid
{
public:
    explicit CWindowAndroid(native_handle_t anw) : m_native(anw), IWindowAndroid(SCreationParams{})
    {
        m_width = ANativeWindow_getWidth(anw);
        m_height = ANativeWindow_getHeight(anw);
    }

    native_handle_t getNativeHandle() const override { return m_native; }
    virtual IClipboardManager* getClipboardManager() { return nullptr; }
    virtual ICursorControl* getCursorControl() { return nullptr; }
private:
    native_handle_t m_native;
};

}
} 

#endif //_NBL_PLATFORM_ANDROID_

#endif
