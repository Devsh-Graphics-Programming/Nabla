#ifndef __NBL_C_WINDOW_ANDROID_H_INCLUDED__
#define __NBL_C_WINDOW_ANDROID_H_INCLUDED__

#include "nbl/ui/IWindowAndroid.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>

namespace nbl::ui
{
class CWindowAndroid : public IWindowAndroid
{
public:
    constexpr static uint32_t CIRCULAR_BUFFER_CAPACITY = 256;
    explicit CWindowAndroid(SCreationParams&& params, native_handle_t anw)
        : m_native(anw), IWindowAndroid(std::move(params))
    {
        m_width = ANativeWindow_getWidth(anw);
        m_height = ANativeWindow_getHeight(anw);
    }

    virtual IClipboardManager* getClipboardManager() { return nullptr; }
    virtual ICursorControl* getCursorControl() { return nullptr; }
    const native_handle_t& getNativeHandle() const override { return m_native; }
    void setCaption(const std::string_view& caption) override {}
    core::map<uint32_t, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannels;
    core::map<uint32_t, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannels;
    bool hasMouseEventChannel(uint32_t deviceId)
    {
        return m_mouseEventChannels.find(deviceId) != m_mouseEventChannels.end();
    }
    bool hasKeyboardEventChannel(uint32_t deviceId)
    {
        return m_keyboardEventChannels.find(deviceId) != m_keyboardEventChannels.end();
    }
    bool addMouseEventChannel(uint32_t deviceId, const core::smart_refctd_ptr<IMouseEventChannel>& channel)
    {
        if(m_mouseEventChannels.find(deviceId) == m_mouseEventChannels.end())
        {
            m_mouseEventChannels.emplace(deviceId, channel);
            return true;
        }
        return false;
    }
    bool addKeyboardEventChannel(uint32_t deviceId, const core::smart_refctd_ptr<IKeyboardEventChannel>& channel)
    {
        if(m_keyboardEventChannels.find(deviceId) == m_keyboardEventChannels.end())
        {
            m_keyboardEventChannels.emplace(deviceId, channel);
            return true;
        }
        return false;
    }
    IMouseEventChannel* getMouseEventChannel(uint32_t deviceId)
    {
        auto ch = m_mouseEventChannels.find(deviceId);
        return m_mouseEventChannels.find(deviceId)->second.get();
    }

    IKeyboardEventChannel* getKeyboardEventChannel(uint32_t deviceId)
    {
        auto ch = m_keyboardEventChannels.find(deviceId);
        return m_keyboardEventChannels.find(deviceId)->second.get();
    }

private:
    native_handle_t m_native;
};

}

#endif  //_NBL_PLATFORM_ANDROID_

#endif
