#include "nbl/ui/CWindowWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{

    LRESULT CALLBACK CWindowWin32::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		CWindowWin32* window = reinterpret_cast<CWindowWin32*>(GetWindowLongPtr(hWnd, GWLP_USERDATA));
		auto* eventCallback = window->getEventCallback();
		switch (message)
		{
		case WM_SHOWWINDOW:
		{
			if (wParam = TRUE)
			{
				eventCallback->onWindowShown(window);
			}
			else
			{
				eventCallback->onWindowHidden(window);
			}
			break;
		}
		case WM_MOVING: [[fallthrough]]
		case WM_MOVE:
		{
			uint32_t newX = (int)LOWORD(lParam); 
			uint32_t newY = (int)HIWORD(lParam);
			eventCallback->onWindowMoved(window, newX, newY);
		}
		case WM_SIZING: [[fallthrough]]
		case WM_SIZE:
		{
			uint32_t newWidth = LOWORD(lParam);
			uint32_t newHeight = HIWORD(lParam);
			eventCallback->onWindowResized(window, newWidth, newHeight);
			switch (wParam)
			{
			case SIZE_MAXIMIZED:
				eventCallback->onWindowMaximized(window);
				break;
			case SIZE_MINIMIZED:
				eventCallback->onWindowMinimized(window);
				break;
			}
			break;
		}
		case WM_SETFOCUS:
		{
			eventCallback->onGainedKeyboardFocus(window);
			break;
		}
		case WM_KILLFOCUS:
		{
			eventCallback->onLostKeyboardFocus(window);
			break;
		}
		case WM_ACTIVATE:
		{
			switch (wParam)
			{
			case WA_CLICKACTIVE:
				eventCallback->onGainedMouseFocus(window);
				break;
			case WA_INACTIVE:
				eventCallback->onLostMouseFocus(window);
			}
			break;
		}
		
		}
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

}
}

#endif