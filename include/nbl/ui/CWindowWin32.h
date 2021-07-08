#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__
#include "nbl/ui/IWindowWin32.h"
<<<<<<< HEAD
#include "nbl_os.h"
=======
#include "os.h"
#include <queue>
>>>>>>> remotes/origin/danylo_system

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
	

<<<<<<< HEAD
private:
    CWindowWin32(uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags) : IWindowWin32(_w, _h, _flags), m_native(NULL)
    {
		// get process handle
		HINSTANCE hinstance = GetModuleHandle(NULL);

		const char* classname = "Nabla Engine";

		WNDCLASSEXA wcex;
		wcex.cbSize = sizeof(WNDCLASSEX);
		wcex.style = CS_HREDRAW | CS_VREDRAW;
		wcex.lpfnWndProc = WndProc;
		wcex.cbClsExtra = 0;
		wcex.cbWndExtra = 0;
		wcex.hInstance = hinstance;
		wcex.hIcon = NULL;
		wcex.hCursor = 0; // LoadCursor(NULL, IDC_ARROW);
		wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
		wcex.lpszMenuName = 0;
		wcex.lpszClassName = classname;
		wcex.hIconSm = 0;

		RegisterClassExA(&wcex);

		// calculate client size

		RECT clientSize;
		clientSize.top = 0;
		clientSize.left = 0;
		clientSize.right = m_width;
		clientSize.bottom = m_height;

		DWORD style = WS_POPUP; // TODO why popup?

		if (!isFullscreen())
		{
			if (!isBorderless())
			{
				style |= WS_BORDER;
				style |= (WS_SYSMENU | WS_CAPTION);
			}

			// ? not sure about those below
			style |= WS_CLIPCHILDREN;
			style |= WS_CLIPSIBLINGS;
		}
		if (isMinimized())
			style |= WS_MINIMIZE;
		if (isMaximized())
			style |= WS_MAXIMIZE;
		// TODO:
		// if (hasMouseCaptured())
		// if (hasInputFocus())
		// if (hasMouseFocus())
		// if (isAlwaysOnTop())

		AdjustWindowRect(&clientSize, style, FALSE);

		const int32_t realWidth = clientSize.right - clientSize.left;
		const int32_t realHeight = clientSize.bottom - clientSize.top;

		int32_t windowLeft = (GetSystemMetrics(SM_CXSCREEN) - realWidth) / 2;
		int32_t windowTop = (GetSystemMetrics(SM_CYSCREEN) - realHeight) / 2;

		if (windowLeft < 0)
			windowLeft = 0;
		if (windowTop < 0)
			windowTop = 0;	// make sure window menus are in screen on creation

		if (isFullscreen())
		{
			windowLeft = 0;
			windowTop = 0;
		}

		// create window

		m_native = CreateWindowA(classname, "", style, windowLeft, windowTop,
			realWidth, realHeight, NULL, NULL, hinstance, NULL);
		m_width = realWidth;
		m_height = realHeight;

		if (!isHidden())
			ShowWindow(m_native, SW_SHOWNORMAL);
		UpdateWindow(m_native);

		// fix ugly ATI driver bugs. Thanks to ariaci
		// TODO still needed?
		MoveWindow(m_native, windowLeft, windowTop, realWidth, realHeight, TRUE);
    }
=======
	// Inherited via IWindowWin32
	virtual IClipboardManager* getClipboardManager() override;
>>>>>>> remotes/origin/danylo_system

};

}
}

#endif

#endif

