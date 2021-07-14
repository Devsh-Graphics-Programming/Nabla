#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__
#include "nbl/ui/IWindowWin32.h"
#include "os.h"
#include <queue>

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
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

	std::map<HANDLE, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannel;
	std::map<HANDLE, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannel;

	void addMouseEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IMouseEventChannel>& channel)
	{
		if (m_mouseEventChannel.find(deviceHandle) == m_mouseEventChannel.end())
			m_mouseEventChannel.emplace(deviceHandle, channel);
	}

	void addKeyboardEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IKeyboardEventChannel>& channel)
	{
		if(m_keyboardEventChannel.find(deviceHandle) == m_keyboardEventChannel.end())
			m_keyboardEventChannel.emplace(deviceHandle, channel);
	}

	core::smart_refctd_ptr<IMouseEventChannel> removeMouseEventChannel(HANDLE deviceHandle)
	{
		RAWINPUT;
		auto it = m_mouseEventChannel.find(deviceHandle);
		auto channel = std::move(it->second);
		m_mouseEventChannel.erase(it);
		return channel;
	}

	core::smart_refctd_ptr<IKeyboardEventChannel> removeKeyboardEventChannel(HANDLE deviceHandle)
	{
		auto it = m_keyboardEventChannel.find(deviceHandle);
		auto channel = std::move(it->second);
		m_keyboardEventChannel.erase(it);
		return channel;
	}
	
	IMouseEventChannel* getMouseEventChannel(HANDLE deviceHandle)
	{
		/** 
		*   This checking is necessary because some devices (like a laptop precision touchpad)
		*   don't get listed in GetRawInputDeviceList but will visible when you get an actual input
		*   from it (the handle to it will be nullptr).
		**/
		auto ch = m_mouseEventChannel.find(deviceHandle);
		if (ch == m_mouseEventChannel.end())
		{
			auto channel = core::make_smart_refctd_ptr<IMouseEventChannel>(CIRCULAR_BUFFER_CAPACITY);
			addMouseEventChannel(deviceHandle, std::move(channel));
		}
		return m_mouseEventChannel.find(deviceHandle)->second.get();
	}

	IKeyboardEventChannel* getKeyboardEventChannel(HANDLE deviceHandle)
	{
		return m_keyboardEventChannel.find(deviceHandle)->second.get();
	}
	
	
	// Inherited via IWindowWin32
	virtual IClipboardManager* getClipboardManager() override;


private:
	static constexpr uint32_t CIRCULAR_BUFFER_CAPACITY = 256;

	void addAlreadyConnectedInputDevices();
	POINT workspaceCoordinatesToScreen(const POINT& p);
};

}
}

#endif

#endif

