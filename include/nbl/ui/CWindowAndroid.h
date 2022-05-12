#ifndef _NBL_C_WINDOW_ANDROID_H_INCLUDED_
#define _NBL_C_WINDOW_ANDROID_H_INCLUDED_

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
		constexpr static inline uint32_t CIRCULAR_BUFFER_CAPACITY = 256;
		explicit inline CWindowAndroid(SCreationParams&& params, native_handle_t anw) : m_native(anw), IWindowAndroid(std::move(params))
		{
			m_width = ANativeWindow_getWidth(anw);
			m_height = ANativeWindow_getHeight(anw);
		}

		virtual inline IClipboardManager* getClipboardManager() override { return nullptr; }
		virtual inline ICursorControl* getCursorControl() override { return nullptr; }

		inline const native_handle_t& getNativeHandle() const override { return m_native; }
		
		inline void setCaption(const std::string_view& caption) override {}

		// WHY THE FUCK ARE THESE PUBLIC?
		core::map<uint32_t, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannels;
		core::map<uint32_t, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannels;

		inline bool hasMouseEventChannel(uint32_t deviceId)
		{
			return m_mouseEventChannels.find(deviceId) != m_mouseEventChannels.end();
		}
		inline bool hasKeyboardEventChannel(uint32_t deviceId)
		{
			return m_keyboardEventChannels.find(deviceId) != m_keyboardEventChannels.end();
		}
		inline bool addMouseEventChannel(uint32_t deviceId, const core::smart_refctd_ptr<IMouseEventChannel>& channel)
		{
			if (m_mouseEventChannels.find(deviceId) == m_mouseEventChannels.end())
			{
				m_mouseEventChannels.emplace(deviceId, channel);
				return true;
			}
			return false;
		}
		inline bool addKeyboardEventChannel(uint32_t deviceId, const core::smart_refctd_ptr<IKeyboardEventChannel>& channel)
		{
			if (m_keyboardEventChannels.find(deviceId) == m_keyboardEventChannels.end())
			{
				m_keyboardEventChannels.emplace(deviceId, channel);
				return true;
			}
			return false;
		}
		inline IMouseEventChannel* getMouseEventChannel(uint32_t deviceId)
		{
			auto ch = m_mouseEventChannels.find(deviceId);
			return m_mouseEventChannels.find(deviceId)->second.get();
		}

		inline IKeyboardEventChannel* getKeyboardEventChannel(uint32_t deviceId)
		{
			auto ch = m_keyboardEventChannels.find(deviceId);
			return m_keyboardEventChannels.find(deviceId)->second.get();
		}

	private:
		native_handle_t m_native;
};

}

#endif //_NBL_PLATFORM_ANDROID_

#endif
