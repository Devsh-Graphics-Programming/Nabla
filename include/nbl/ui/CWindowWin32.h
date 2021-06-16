#ifndef __C_WINDOW_WIN32_H_INCLUDED__
#define __C_WINDOW_WIN32_H_INCLUDED__

#include "nbl/ui/IWindowWin32.h"
#include "nbl_os.h"

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{

class CWindowWin32 final : public IWindowWin32
{
	// TODO design for event listener etc, WndProc will somehow pass events to provided event listener
	static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

public:
	explicit CWindowWin32(native_handle_t hwnd) : m_native(hwnd)
	{
		RECT rect;
		GetWindowRect(hwnd, &rect);

		m_width = rect.right - rect.left;
		m_height = rect.bottom - rect.top;

		// TODO m_flags
	}

	native_handle_t getNativeHandle() const override { return m_native; }

	static core::smart_refctd_ptr<CWindowWin32> create(uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		if ((_flags & (ECF_MINIMIZED | ECF_MAXIMIZED)) == (ECF_MINIMIZED | ECF_MAXIMIZED))
			return nullptr;

		CWindowWin32* win = new CWindowWin32(_w, _h, _flags);
		return core::smart_refctd_ptr<CWindowWin32>(win, core::dont_grab);
	}

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

    native_handle_t m_native;
};

}
}

#endif

#endif
