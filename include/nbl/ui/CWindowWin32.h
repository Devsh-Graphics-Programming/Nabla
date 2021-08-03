#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__
#include "nbl/ui/IWindowWin32.h"
#include <queue>

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl::ui
{

class CWindowManagerWin32;

class CWindowWin32 final : public IWindowWin32
{
public:
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	static E_KEY_CODE getNablaKeyCodeFromNative(uint8_t nativeWindowsKeyCode);

	CWindowWin32(CWindowManagerWin32* winManager, SCreationParams&& params, native_handle_t hwnd);

	native_handle_t getNativeHandle() const override { return m_native; }

	~CWindowWin32() override;
private:
	CWindowWin32(CWindowManagerWin32* winManager, core::smart_refctd_ptr<system::ISystem>&& sys, SCreationParams&& params);

    native_handle_t m_native;

	CWindowManagerWin32* m_windowManager;

	core::map<HANDLE, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannels;
	core::map<HANDLE, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannels;
	
	/* 
	*  Storing this data is required for the device removal to work properly
	*  When you get a message about the device removal, its type isn't accessible anymore.
	*  When adding new devices, we return if we didnt have the device in the list before.
	*/
	core::map<HANDLE, uint32_t> m_deviceTypes;
	bool addMouseEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IMouseEventChannel>& channel)
	{
		if (m_mouseEventChannels.find(deviceHandle) == m_mouseEventChannels.end())
		{
			m_mouseEventChannels.emplace(deviceHandle, channel);
			m_deviceTypes.emplace(deviceHandle, RIM_TYPEMOUSE);
			return true;
		}
		return false;
	}
	bool addKeyboardEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IKeyboardEventChannel>& channel)
	{
		if (m_keyboardEventChannels.find(deviceHandle) == m_keyboardEventChannels.end())
		{
			m_keyboardEventChannels.emplace(deviceHandle, channel);
			m_deviceTypes.emplace(deviceHandle, RIM_TYPEKEYBOARD);
			return true;
		}
		return false;
	}

	core::smart_refctd_ptr<IMouseEventChannel> removeMouseEventChannel(HANDLE deviceHandle)
	{
		RAWINPUT;
		auto it = m_mouseEventChannels.find(deviceHandle);
		auto channel = std::move(it->second);
		m_mouseEventChannels.erase(it);
		m_deviceTypes.erase(m_deviceTypes.find(deviceHandle));
		return channel;
	}

	core::smart_refctd_ptr<IKeyboardEventChannel> removeKeyboardEventChannel(HANDLE deviceHandle)
	{
		auto it = m_keyboardEventChannels.find(deviceHandle);
		auto channel = std::move(it->second);
		m_keyboardEventChannels.erase(it);
		m_deviceTypes.erase(m_deviceTypes.find(deviceHandle));
		return channel;
	}

	int32_t getDeviceType(HANDLE h)
	{
		auto type = m_deviceTypes.find(h);
		if (type != m_deviceTypes.end()) return type->second;
		return -1;
	}
	
	IMouseEventChannel* getMouseEventChannel(HANDLE deviceHandle)
	{
		/** 
		*   This checking is necessary because some devices (like a laptop precision touchpad)
		*   don't get listed in GetRawInputDeviceList but will visible when you get an actual input
		*   from it (the handle to it will be nullptr).
		**/
		auto ch = m_mouseEventChannels.find(deviceHandle);
		if (ch == m_mouseEventChannels.end())
		{
			auto channel = core::make_smart_refctd_ptr<IMouseEventChannel>(CIRCULAR_BUFFER_CAPACITY);
			addMouseEventChannel(deviceHandle, std::move(channel));
		}
		return m_mouseEventChannels.find(deviceHandle)->second.get();
	}

	IKeyboardEventChannel* getKeyboardEventChannel(HANDLE deviceHandle)
	{
		return m_keyboardEventChannels.find(deviceHandle)->second.get();
	}
	
	
	// Inherited via IWindowWin32
	virtual IClipboardManager* getClipboardManager() override;


private:
	static constexpr uint32_t CIRCULAR_BUFFER_CAPACITY = 256;

	void addAlreadyConnectedInputDevices();
	POINT workspaceCoordinatesToScreen(const POINT& p);
};

}

#endif

#endif

