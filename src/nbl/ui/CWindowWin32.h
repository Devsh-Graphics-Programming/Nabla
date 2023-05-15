#ifndef _NBL_UI_C_WINDOW_WIN32_H_INCLUDED_
#define _NBL_UI_C_WINDOW_WIN32_H_INCLUDED_


#include "nbl/ui/CWindowManagerWin32.h"
#include "nbl/ui/CClipboardManagerWin32.h"

#include <cstdint>
#include <queue>


#ifdef _NBL_PLATFORM_WINDOWS_
namespace nbl::ui
{

class NBL_API2 CWindowWin32 final : public IWindowWin32
{
	public:

		CWindowWin32(SCreationParams&& params, core::smart_refctd_ptr<CWindowManagerWin32>&& winManager, native_handle_t hwnd);

		inline const native_handle_t& getNativeHandle() const override {return m_native;}
		
		inline void setCaption(const std::string_view& caption) override
		{
			SetWindowText(m_native,caption.data());
		}

		inline IClipboardManager* getClipboardManager() override
		{
			return m_clipboardManager.get();
		}

		inline ICursorControl* getCursorControl() override {return m_windowManager.get();}

		inline IWindowManager* getManager() const override {return m_windowManager.get();}

		//!
		static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

	protected:
		inline ~CWindowWin32() override
		{
			m_windowManager->destroyWindow(this);
		}

	private:
		core::smart_refctd_ptr<CWindowManagerWin32> m_windowManager;
		native_handle_t m_native;
		core::smart_refctd_ptr<CClipboardManagerWin32> m_clipboardManager;

		core::map<HANDLE, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannels;
		core::map<HANDLE, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannels;

		/* 
		*  Storing this data is required for the device removal to work properly
		*  When you get a message about the device removal, its type isn't accessible anymore.
		*  When adding new devices, we return if we didnt have the device in the list before.
		*/
		core::map<HANDLE,uint32_t> m_deviceTypes;
		inline bool addMouseEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IMouseEventChannel>& channel)
		{
			if (m_mouseEventChannels.find(deviceHandle) == m_mouseEventChannels.end())
			{
				m_mouseEventChannels.emplace(deviceHandle, channel);
				m_deviceTypes.emplace(deviceHandle, RIM_TYPEMOUSE);
				return true;
			}
			return false;
		}
		inline bool addKeyboardEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IKeyboardEventChannel>& channel)
		{
			if (m_keyboardEventChannels.find(deviceHandle) == m_keyboardEventChannels.end())
			{
				m_keyboardEventChannels.emplace(deviceHandle, channel);
				m_deviceTypes.emplace(deviceHandle, RIM_TYPEKEYBOARD);
				return true;
			}
			return false;
		}

		inline core::smart_refctd_ptr<IMouseEventChannel> removeMouseEventChannel(HANDLE deviceHandle)
		{
			RAWINPUT;
			auto it = m_mouseEventChannels.find(deviceHandle);
			auto channel = std::move(it->second);
			m_mouseEventChannels.erase(it);
			m_deviceTypes.erase(m_deviceTypes.find(deviceHandle));
			return channel;
		}

		inline core::smart_refctd_ptr<IKeyboardEventChannel> removeKeyboardEventChannel(HANDLE deviceHandle)
		{
			auto it = m_keyboardEventChannels.find(deviceHandle);
			auto channel = std::move(it->second);
			m_keyboardEventChannels.erase(it);
			m_deviceTypes.erase(m_deviceTypes.find(deviceHandle));
			return channel;
		}

		inline int32_t getDeviceType(HANDLE h)
		{
			auto type = m_deviceTypes.find(h);
			if (type != m_deviceTypes.end()) return type->second;
			return -1;
		}

		static constexpr inline uint32_t ChannelEventCapacity = 256;
		inline IMouseEventChannel* getMouseEventChannel(HANDLE deviceHandle)
		{
			/** 
			*   This checking is necessary because some devices (like a laptop precision touchpad)
			*   don't get listed in GetRawInputDeviceList but will visible when you get an actual input
			*   from it (the handle to it will be nullptr).
			**/
			auto ch = m_mouseEventChannels.find(deviceHandle);
			// windows is a special boy
			if (ch==m_mouseEventChannels.end())
			{
				auto channel = core::make_smart_refctd_ptr<IMouseEventChannel>(ChannelEventCapacity);
				if (addMouseEventChannel(deviceHandle,std::move(channel)))
					m_cb->onMouseConnected(this,std::move(channel));
			}
			return m_mouseEventChannels.find(deviceHandle)->second.get();
		}
		inline IKeyboardEventChannel* getKeyboardEventChannel(HANDLE deviceHandle)
		{
			auto ch = m_keyboardEventChannels.find(deviceHandle);
			// anydesk makes windows a special boy
			if (ch==m_keyboardEventChannels.end())
			{
				auto channel = core::make_smart_refctd_ptr<IKeyboardEventChannel>(ChannelEventCapacity);
				if (addKeyboardEventChannel(deviceHandle, std::move(channel)))
					m_cb->onKeyboardConnected(this, std::move(channel));
			}
			return m_keyboardEventChannels.find(deviceHandle)->second.get();
		}

		POINT workspaceCoordinatesToScreen(const POINT& p);
};

}
#endif

#endif

