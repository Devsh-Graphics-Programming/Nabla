#include "nbl/ui/CWindowWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{

    LRESULT CALLBACK CWindowWin32::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
	{
		os::Printer::log("CWindowWin32 event listening unimplemented!", ELL_ERROR);
		/*
#ifndef WM_MOUSEWHEEL
#define WM_MOUSEWHEEL 0x020A
#endif
#ifndef WHEEL_DELTA
#define WHEEL_DELTA 120
#endif

		nbl::CIrrDeviceWin32* dev = 0;
		nbl::SEvent event;

		static int32_t ClickCount = 0;
		if (GetCapture() != hWnd && ClickCount > 0)
			ClickCount = 0;


		struct messageMap
		{
			int32_t group;
			UINT winMessage;
			int32_t irrMessage;
		};

		static messageMap mouseMap[] =
		{
			{0, WM_LBUTTONDOWN, nbl::EMIE_LMOUSE_PRESSED_DOWN},
			{1, WM_LBUTTONUP,   nbl::EMIE_LMOUSE_LEFT_UP},
			{0, WM_RBUTTONDOWN, nbl::EMIE_RMOUSE_PRESSED_DOWN},
			{1, WM_RBUTTONUP,   nbl::EMIE_RMOUSE_LEFT_UP},
			{0, WM_MBUTTONDOWN, nbl::EMIE_MMOUSE_PRESSED_DOWN},
			{1, WM_MBUTTONUP,   nbl::EMIE_MMOUSE_LEFT_UP},
			{2, WM_MOUSEMOVE,   nbl::EMIE_MOUSE_MOVED},
			{3, WM_MOUSEWHEEL,  nbl::EMIE_MOUSE_WHEEL},
			{-1, 0, 0}
		};

		// handle grouped events
		messageMap* m = mouseMap;
		while (m->group >= 0 && m->winMessage != message)
			m += 1;

		if (m->group >= 0)
		{
			if (m->group == 0)	// down
			{
				ClickCount++;
				SetCapture(hWnd);
			}
			else
				if (m->group == 1)	// up
				{
					ClickCount--;
					if (ClickCount < 1)
					{
						ClickCount = 0;
						ReleaseCapture();
					}
				}

			event.EventType = nbl::EET_MOUSE_INPUT_EVENT;
			event.MouseInput.Event = (nbl::EMOUSE_INPUT_EVENT) m->irrMessage;
			event.MouseInput.X = (short)LOWORD(lParam);
			event.MouseInput.Y = (short)HIWORD(lParam);
			event.MouseInput.Shift = ((LOWORD(wParam) & MK_SHIFT) != 0);
			event.MouseInput.Control = ((LOWORD(wParam) & MK_CONTROL) != 0);
			// left and right mouse buttons
			event.MouseInput.ButtonStates = wParam & (MK_LBUTTON | MK_RBUTTON);
			// middle and extra buttons
			if (wParam & MK_MBUTTON)
				event.MouseInput.ButtonStates |= nbl::EMBSM_MIDDLE;
#if(_WIN32_WINNT >= 0x0500)
			if (wParam & MK_XBUTTON1)
				event.MouseInput.ButtonStates |= nbl::EMBSM_EXTRA1;
			if (wParam & MK_XBUTTON2)
				event.MouseInput.ButtonStates |= nbl::EMBSM_EXTRA2;
#endif
			event.MouseInput.Wheel = 0.f;

			// wheel
			if (m->group == 3)
			{
				POINT p; // fixed by jox
				p.x = 0; p.y = 0;
				ClientToScreen(hWnd, &p);
				event.MouseInput.X -= p.x;
				event.MouseInput.Y -= p.y;
				event.MouseInput.Wheel = ((float)((short)HIWORD(wParam))) / (float)WHEEL_DELTA;
			}

			dev = getDeviceFromHWnd(hWnd);
			if (dev)
			{
				dev->postEventFromUser(event);

				if (event.MouseInput.Event >= nbl::EMIE_LMOUSE_PRESSED_DOWN && event.MouseInput.Event <= nbl::EMIE_MMOUSE_PRESSED_DOWN)
				{
					uint32_t clicks = dev->checkSuccessiveClicks(event.MouseInput.X, event.MouseInput.Y, event.MouseInput.Event);
					if (clicks == 2)
					{
						event.MouseInput.Event = (nbl::EMOUSE_INPUT_EVENT)(nbl::EMIE_LMOUSE_DOUBLE_CLICK + event.MouseInput.Event - nbl::EMIE_LMOUSE_PRESSED_DOWN);
						dev->postEventFromUser(event);
					}
					else if (clicks == 3)
					{
						event.MouseInput.Event = (nbl::EMOUSE_INPUT_EVENT)(nbl::EMIE_LMOUSE_TRIPLE_CLICK + event.MouseInput.Event - nbl::EMIE_LMOUSE_PRESSED_DOWN);
						dev->postEventFromUser(event);
					}
				}
			}
			return 0;
		}

		switch (message)
		{
		case WM_PAINT:
		{
			PAINTSTRUCT ps;
			BeginPaint(hWnd, &ps);
			EndPaint(hWnd, &ps);
		}
		return 0;

		case WM_ERASEBKGND:
			return 0;

		case WM_SYSKEYDOWN:
		case WM_SYSKEYUP:
		case WM_KEYDOWN:
		case WM_KEYUP:
		{
			BYTE allKeys[256];

			event.EventType = nbl::EET_KEY_INPUT_EVENT;
			event.KeyInput.Key = (nbl::EKEY_CODE)wParam;
			event.KeyInput.PressedDown = (message == WM_KEYDOWN || message == WM_SYSKEYDOWN);

			const UINT MY_MAPVK_VSC_TO_VK_EX = 3; // MAPVK_VSC_TO_VK_EX should be in SDK according to MSDN, but isn't in mine.
			if (event.KeyInput.Key == nbl::KEY_SHIFT)
			{
				// this will fail on systems before windows NT/2000/XP, not sure _what_ will return there instead.
				event.KeyInput.Key = (nbl::EKEY_CODE)MapVirtualKey(((lParam >> 16) & 255), MY_MAPVK_VSC_TO_VK_EX);
			}
			if (event.KeyInput.Key == nbl::KEY_CONTROL)
			{
				event.KeyInput.Key = (nbl::EKEY_CODE)MapVirtualKey(((lParam >> 16) & 255), MY_MAPVK_VSC_TO_VK_EX);
				// some keyboards will just return LEFT for both - left and right keys. So also check extend bit.
				if (lParam & 0x1000000)
					event.KeyInput.Key = nbl::KEY_RCONTROL;
			}
			if (event.KeyInput.Key == nbl::KEY_MENU)
			{
				event.KeyInput.Key = (nbl::EKEY_CODE)MapVirtualKey(((lParam >> 16) & 255), MY_MAPVK_VSC_TO_VK_EX);
				if (lParam & 0x1000000)
					event.KeyInput.Key = nbl::KEY_RMENU;
			}

			GetKeyboardState(allKeys);

			event.KeyInput.Shift = ((allKeys[VK_SHIFT] & 0x80) != 0);
			event.KeyInput.Control = ((allKeys[VK_CONTROL] & 0x80) != 0);

			// Handle unicode and deadkeys in a way that works since Windows 95 and nt4.0
			// Using ToUnicode instead would be shorter, but would to my knowledge not run on 95 and 98.
			WORD keyChars[2];
			UINT scanCode = HIWORD(lParam);
			int conversionResult = ToAsciiEx(wParam, scanCode, allKeys, keyChars, 0, KEYBOARD_INPUT_HKL);
			if (conversionResult == 1)
			{
				WORD unicodeChar;
				MultiByteToWideChar(
					KEYBOARD_INPUT_CODEPAGE,
					MB_PRECOMPOSED, // default
					(LPCSTR)keyChars,
					sizeof(keyChars),
					(WCHAR*)&unicodeChar,
					1);
				event.KeyInput.Char = unicodeChar;
			}
			else
				event.KeyInput.Char = 0;

			// allow composing characters like '@' with Alt Gr on non-US keyboards
			if ((allKeys[VK_MENU] & 0x80) != 0)
				event.KeyInput.Control = 0;

			dev = getDeviceFromHWnd(hWnd);
			if (dev)
				dev->postEventFromUser(event);

			if (message == WM_SYSKEYDOWN || message == WM_SYSKEYUP)
				return DefWindowProc(hWnd, message, wParam, lParam);
			else
				return 0;
		}

		case WM_SIZE:
		{
			// resize
			dev = getDeviceFromHWnd(hWnd);
			if (dev)
				dev->OnResized();
		}
		return 0;

		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;

		case WM_SYSCOMMAND:
			// prevent screensaver or monitor powersave mode from starting
			if ((wParam & 0xFFF0) == SC_SCREENSAVE ||
				(wParam & 0xFFF0) == SC_MONITORPOWER ||
				(wParam & 0xFFF0) == SC_KEYMENU
				)
				return 0;

			break;

		case WM_ACTIVATE:
			// we need to take care for screen changes, e.g. Alt-Tab
			dev = getDeviceFromHWnd(hWnd);
			if (dev && dev->isFullscreen())
			{
				if ((wParam & 0xFF) == WA_INACTIVE)
				{
					// If losing focus we minimize the app to show other one
					ShowWindow(hWnd, SW_MINIMIZE);
					// and switch back to default resolution
					dev->switchToFullscreen(true);
				}
				else
				{
					// Otherwise we retore the fullscreen Irrlicht app
					SetForegroundWindow(hWnd);
					ShowWindow(hWnd, SW_RESTORE);
					// and set the fullscreen resolution again
					dev->switchToFullscreen();
				}
			}
			break;

		case WM_USER:
			event.EventType = nbl::EET_USER_EVENT;
			event.UserEvent.UserData1 = (int32_t)wParam;
			event.UserEvent.UserData2 = (int32_t)lParam;
			dev = getDeviceFromHWnd(hWnd);

			if (dev)
				dev->postEventFromUser(event);

			return 0;

		case WM_SETCURSOR:
			// because Windows forgot about that in the meantime
			dev = getDeviceFromHWnd(hWnd);
			if (dev)
			{
				dev->getCursorControl()->setActiveIcon(dev->getCursorControl()->getActiveIcon());
				dev->getCursorControl()->setVisible(dev->getCursorControl()->isVisible());
			}
			break;

		case WM_INPUTLANGCHANGE:
			// get the new codepage used for keyboard input
			KEYBOARD_INPUT_HKL = GetKeyboardLayout(0);
			KEYBOARD_INPUT_CODEPAGE = LocaleIdToCodepage(LOWORD(KEYBOARD_INPUT_HKL));
			return 0;
		}
		*/
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

}
}

#endif