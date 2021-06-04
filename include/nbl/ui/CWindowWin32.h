#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__
#include "nbl/ui/IWindowWin32.h"
#include "os.h"
#include <queue>
#include <hidpi.h>

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{

class CWindowWin32 final : public IWindowWin32
{
public:
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);
	static E_KEY_CODE getNablaKeyCodeFromNative(uint32_t nativeWindowsKeyCode);

	//TODO
	explicit CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, native_handle_t hwnd) : IWindowWin32(std::move(sys))
	{
		assert(false);
		RECT rect;
		GetWindowRect(hwnd, &rect);

		m_width = rect.right - rect.left;
		m_height = rect.bottom - rect.top;
		// TODO m_flags
	}

	native_handle_t getNativeHandle() const override { return m_native; }

	static core::smart_refctd_ptr<CWindowWin32> create(core::smart_refctd_ptr<system::ISystem>&& sys, int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowWin32* win = new CWindowWin32(std::move(sys), _x, _y, _w, _h, _flags);
		return core::smart_refctd_ptr<CWindowWin32>(win, core::dont_grab);
	}

	~CWindowWin32() override;
private:
	CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags);

    native_handle_t m_native;

	std::map<HANDLE, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannel;
	std::map<HANDLE, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannel;

	void addMouseEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IMouseEventChannel>& channel)
	{
		assert(m_mouseEventChannel.find(deviceHandle) == m_mouseEventChannel.end());
		m_mouseEventChannel.emplace(deviceHandle, channel);
	}

	void addKeyboardEventChannel(HANDLE deviceHandle, const core::smart_refctd_ptr<IKeyboardEventChannel>& channel)
	{
		assert(m_keyboardEventChannel.find(deviceHandle) == m_keyboardEventChannel.end());
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
		return m_mouseEventChannel.find(deviceHandle)->second.get();
	}

	IKeyboardEventChannel* getKeyboardEventChannel(HANDLE deviceHandle)
	{
		return m_keyboardEventChannel.find(deviceHandle)->second.get();
	}
	
};

}
}

#endif

#endif

#endif