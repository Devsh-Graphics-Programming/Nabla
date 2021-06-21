
#ifdef _NBL_PLATFORM_WINDOWS_
#include "nbl/ui/CWindowWin32.h"
#include <hidusage.h>
#include <codecvt>

namespace nbl {
namespace ui
{

	CWindowWin32::CWindowWin32(core::smart_refctd_ptr<system::ISystem>&& sys, int _x, int _y, uint32_t _w, uint32_t _h, E_CREATE_FLAGS _flags)
	{
		m_width = _w; m_height = _h;
		windowThreadHandler.createWindow(_x, _y, _w, _h, _flags, m_native);
		SetWindowLongPtr(m_native, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
	}

	CWindowWin32::~CWindowWin32()
	{
		windowThreadHandler.destroyWindow(m_native);
	}

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
		case WM_MOVING: [[fallthrough]];
		case WM_MOVE:
		{
			int newX = (int)LOWORD(lParam);
			int newY = (int)HIWORD(lParam);
			eventCallback->onWindowMoved(window, newX, newY);
		}
		case WM_SIZING: [[fallthrough]];
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
				break;
			}
			break;
		}
		case WM_INPUT_DEVICE_CHANGE:
		{
			constexpr uint32_t CIRCULAR_BUFFER_CAPACITY = 256;
			RID_DEVICE_INFO deviceInfo;
			deviceInfo.cbSize = sizeof(RID_DEVICE_INFO);
			UINT size = sizeof(RID_DEVICE_INFO);
			GetRawInputDeviceInfoA((HANDLE)lParam, RIDI_DEVICEINFO, &deviceInfo, &size);

			HANDLE deviceHandle = HANDLE(lParam);


			switch (wParam)
			{
			case GIDC_ARRIVAL:
			{
				if (deviceInfo.dwType == RIM_TYPEMOUSE)
				{
					auto channel = core::make_smart_refctd_ptr<IMouseEventChannel>(CIRCULAR_BUFFER_CAPACITY);
					eventCallback->onMouseConnected(window, core::smart_refctd_ptr<IMouseEventChannel>(channel));
					window->addMouseEventChannel(deviceHandle, std::move(channel));
				}
				else if (deviceInfo.dwType == RIM_TYPEKEYBOARD)
				{
					auto channel = core::make_smart_refctd_ptr<IKeyboardEventChannel>(CIRCULAR_BUFFER_CAPACITY);
					eventCallback->onKeyboardConnected(window, core::smart_refctd_ptr<IKeyboardEventChannel>(channel));
					window->addKeyboardEventChannel(deviceHandle, std::move(channel));
				}
				else if (deviceInfo.dwType == RIM_TYPEHID)
				{
					// TODO 
				}
				break;
			}
			case GIDC_REMOVAL:
			{
				if (deviceInfo.dwType == RIM_TYPEMOUSE)
				{
					auto channel = window->removeMouseEventChannel(deviceHandle);
					eventCallback->onMouseDisconnected(window, channel.get());
				}
				else if (deviceInfo.dwType == RIM_TYPEKEYBOARD)
				{
					auto channel = window->removeKeyboardEventChannel(deviceHandle);
					eventCallback->onKeyboardDisconnected(window, channel.get());
				}
				else if (deviceInfo.dwType == RIM_TYPEHID)
				{
					// TODO 
				}
				break;
			}
			}
			break;
		}
		case WM_INPUT:
		{
			RAWINPUT rawInput;
			UINT size;
			GetRawInputData((HRAWINPUT)lParam, RID_HEADER, &rawInput, &size, sizeof rawInput.header);
			GetRawInputData((HRAWINPUT)lParam, RID_INPUT, &rawInput, &size, sizeof rawInput.header);
			HANDLE device = rawInput.header.hDevice;
			switch (rawInput.header.dwType)
			{
			case RIM_TYPEMOUSE:
			{
				auto* inputChannel = window->getMouseEventChannel(device);
				RAWMOUSE rawMouse = rawInput.data.mouse;
 
				if ((rawMouse.usFlags & MOUSE_MOVE_RELATIVE) == MOUSE_MOVE_RELATIVE)
				{
					assert(rawMouse.lLastX != 0 || rawMouse.lLastY != 0);
					SMouseEvent event;
					event.type = SMouseEvent::EET_MOVEMENT;
					event.movementEvent.movementX = rawMouse.lLastX;
					event.movementEvent.movementY = rawMouse.lLastY;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				if (rawMouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EET_CLICK;
					event.clickEvent.mouseButton = E_MOUSE_BUTTON::EMB_LEFT_BUTTON;
					event.clickEvent.action = SMouseEvent::SClickEvent::EA_PRESSED;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EET_CLICK;
					event.clickEvent.mouseButton = E_MOUSE_BUTTON::EMB_LEFT_BUTTON;
					event.clickEvent.action = SMouseEvent::SClickEvent::EA_RELEASED;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				if (rawMouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EET_CLICK;
					event.clickEvent.mouseButton = E_MOUSE_BUTTON::EMB_RIGHT_BUTTON;
					event.clickEvent.action = SMouseEvent::SClickEvent::EA_PRESSED;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EET_CLICK;
					event.clickEvent.mouseButton = E_MOUSE_BUTTON::EMB_RIGHT_BUTTON;
					event.clickEvent.action = SMouseEvent::SClickEvent::EA_RELEASED;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				if (rawMouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EET_CLICK;
					event.clickEvent.mouseButton = E_MOUSE_BUTTON::EMB_MIDDLE_BUTTON;
					event.clickEvent.action = SMouseEvent::SClickEvent::EA_PRESSED;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EET_CLICK;
					event.clickEvent.mouseButton = E_MOUSE_BUTTON::EMB_MIDDLE_BUTTON;
					event.clickEvent.action = SMouseEvent::SClickEvent::EA_RELEASED;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				// TODO other mouse buttons

				if (rawMouse.usButtonFlags & RI_MOUSE_WHEEL)
				{
					SHORT wheelDelta = static_cast<SHORT>(rawMouse.usButtonData);
					SMouseEvent event;
					event.type = SMouseEvent::EET_SCROLL;
					event.scrollEvent.verticalScroll = wheelDelta;
					event.scrollEvent.horizontalScroll = 0;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_HWHEEL)
				{
					SHORT wheelDelta = static_cast<SHORT>(rawMouse.usButtonData);
					SMouseEvent event;
					event.type = SMouseEvent::EET_SCROLL;
					event.scrollEvent.verticalScroll = 0;
					event.scrollEvent.horizontalScroll = wheelDelta;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				break;
			}
			case RIM_TYPEKEYBOARD:
			{
				auto inputChannel = window->getKeyboardEventChannel(device);
				RAWKEYBOARD rawKeyboard = rawInput.data.keyboard;
				switch (rawKeyboard.Message)
				{
				case WM_KEYDOWN: [[fallthrough]];
				case WM_SYSKEYDOWN:
				{
					SKeyboardEvent event;
					event.action = SKeyboardEvent::ECA_PRESSED;
					event.window = window;
					event.keyCode = getNablaKeyCodeFromNative(rawKeyboard.VKey);
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				case WM_KEYUP: [[fallthrough]];
				case WM_SYSKEYUP:
				{
					SKeyboardEvent event;
					event.action = SKeyboardEvent::ECA_RELEASED;
					event.window = window;
					event.keyCode = getNablaKeyCodeFromNative(rawKeyboard.VKey);
					auto lk = inputChannel->lockBackgroundBuffer();
					inputChannel->pushIntoBackground(std::move(event));
				}
				}

				break;
			}
			case RIM_TYPEHID:
			{
				// TODO
				break;
			}
			}
			break;
		}
		}
		return DefWindowProc(hWnd, message, wParam, lParam);
	}

	E_KEY_CODE CWindowWin32::getNablaKeyCodeFromNative(uint8_t nativeWindowsKeyCode)
	{
		nbl::ui::E_KEY_CODE nablaKeyCode = EKC_NONE;
		switch (nativeWindowsKeyCode)
		{
		case VK_BACK:			nablaKeyCode = EKC_BACKSPACE; break;
		case VK_TAB:			nablaKeyCode = EKC_TAB; break;
		case VK_CLEAR:			nablaKeyCode = EKC_CLEAR; break;
		case VK_RETURN:			nablaKeyCode = EKC_ENTER; break;
		case VK_SHIFT:			[[fallthrough]];
		case VK_LSHIFT:			nablaKeyCode = EKC_LEFT_SHIFT; break;
		case VK_RSHIFT:			nablaKeyCode = EKC_RIGHT_SHIFT; break;
		case VK_CONTROL:		[[fallthrough]];
		case VK_LCONTROL:		nablaKeyCode = EKC_LEFT_CONTROL; break;
		case VK_RCONTROL:		nablaKeyCode = EKC_RIGHT_CONTROL; break;
		case VK_LMENU:			[[fallthrough]];
		case VK_MENU:			nablaKeyCode = EKC_LEFT_ALT; break;
		case VK_RMENU:			nablaKeyCode = EKC_RIGHT_ALT; break;
		case VK_PAUSE:			nablaKeyCode = EKC_PAUSE; break;
		case VK_CAPITAL:		nablaKeyCode = EKC_CAPS_LOCK; break;
		case VK_ESCAPE:			nablaKeyCode = EKC_ESCAPE; break;
		case VK_SPACE:			nablaKeyCode = EKC_SPACE; break;
		case VK_PRIOR:			nablaKeyCode = EKC_PAGE_UP; break;
		case VK_NEXT:			nablaKeyCode = EKC_PAGE_DOWN; break;
		case VK_END:			nablaKeyCode = EKC_END; break;
		case VK_HOME:			nablaKeyCode = EKC_HOME; break;
		case VK_LEFT:			nablaKeyCode = EKC_LEFT_ARROW; break;
		case VK_RIGHT:			nablaKeyCode = EKC_RIGHT_ARROW; break;
		case VK_UP:				nablaKeyCode = EKC_UP_ARROW; break;
		case VK_DOWN:			nablaKeyCode = EKC_DOWN_ARROW; break;
		case VK_SELECT:			nablaKeyCode = EKC_SELECT; break;
		case VK_PRINT:			nablaKeyCode = EKC_PRINT; break;
		case VK_EXECUTE:		nablaKeyCode = EKC_EXECUTE; break;
		case VK_SNAPSHOT:		nablaKeyCode = EKC_PRINT_SCREEN; break;
		case VK_INSERT:			nablaKeyCode = EKC_INSERT; break;
		case VK_DELETE:			nablaKeyCode = EKC_DELETE; break;
		case VK_HELP:			nablaKeyCode = EKC_HELP; break;
		case '0':				nablaKeyCode = EKC_0; break;
		case '1':				nablaKeyCode = EKC_1; break;
		case '2':				nablaKeyCode = EKC_2; break;
		case '3':				nablaKeyCode = EKC_3; break;
		case '4':				nablaKeyCode = EKC_4; break;
		case '5':				nablaKeyCode = EKC_5; break;
		case '6':				nablaKeyCode = EKC_6; break;
		case '7':				nablaKeyCode = EKC_7; break;
		case '8':				nablaKeyCode = EKC_8; break;
		case '9':				nablaKeyCode = EKC_9; break;
		case VK_NUMPAD0:		nablaKeyCode = EKC_NUMPAD_0; break;
		case VK_NUMPAD1:		nablaKeyCode = EKC_NUMPAD_1; break;
		case VK_NUMPAD2:		nablaKeyCode = EKC_NUMPAD_2; break;
		case VK_NUMPAD3:		nablaKeyCode = EKC_NUMPAD_3; break;
		case VK_NUMPAD4:		nablaKeyCode = EKC_NUMPAD_4; break;
		case VK_NUMPAD5:		nablaKeyCode = EKC_NUMPAD_5; break;
		case VK_NUMPAD6:		nablaKeyCode = EKC_NUMPAD_6; break;
		case VK_NUMPAD7:		nablaKeyCode = EKC_NUMPAD_7; break;
		case VK_NUMPAD8:		nablaKeyCode = EKC_NUMPAD_8; break;
		case VK_NUMPAD9:		nablaKeyCode = EKC_NUMPAD_9; break;
		case 'A':				nablaKeyCode = EKC_A; break;
		case 'B':				nablaKeyCode = EKC_B; break;
		case 'C':				nablaKeyCode = EKC_C; break;
		case 'D':				nablaKeyCode = EKC_D; break;
		case 'E':				nablaKeyCode = EKC_E; break;
		case 'F':				nablaKeyCode = EKC_F; break;
		case 'G':				nablaKeyCode = EKC_G; break;
		case 'H':				nablaKeyCode = EKC_H; break;
		case 'I':				nablaKeyCode = EKC_I; break;
		case 'J':				nablaKeyCode = EKC_J; break;
		case 'K':				nablaKeyCode = EKC_K; break;
		case 'L':				nablaKeyCode = EKC_L; break;
		case 'M':				nablaKeyCode = EKC_M; break;
		case 'N':				nablaKeyCode = EKC_N; break;
		case 'O':				nablaKeyCode = EKC_O; break;
		case 'P':				nablaKeyCode = EKC_P; break;
		case 'Q':				nablaKeyCode = EKC_Q; break;
		case 'R':				nablaKeyCode = EKC_R; break;
		case 'S':				nablaKeyCode = EKC_S; break;
		case 'T':				nablaKeyCode = EKC_T; break;
		case 'U':				nablaKeyCode = EKC_U; break;
		case 'V':				nablaKeyCode = EKC_V; break;
		case 'W':				nablaKeyCode = EKC_W; break;
		case 'X':				nablaKeyCode = EKC_X; break;
		case 'Y':				nablaKeyCode = EKC_Y; break;
		case 'Z':				nablaKeyCode = EKC_Z; break;
		case VK_LWIN:			nablaKeyCode = EKC_LEFT_WIN; break;
		case VK_RWIN:			nablaKeyCode = EKC_RIGHT_WIN; break;
		case VK_APPS:			nablaKeyCode = EKC_APPS; break;
		case VK_ADD:			nablaKeyCode = EKC_ADD; break;
		case VK_SUBTRACT:		nablaKeyCode = EKC_SUBTRACT; break;
		case VK_MULTIPLY:		nablaKeyCode = EKC_MULTIPLY; break;
		case VK_DIVIDE:			nablaKeyCode = EKC_DIVIDE; break;
		case VK_SEPARATOR:		nablaKeyCode = EKC_SEPARATOR; break;
		case VK_NUMLOCK:		nablaKeyCode = EKC_NUM_LOCK; break;
		case VK_SCROLL:			nablaKeyCode = EKC_SCROLL_LOCK; break;
		case VK_VOLUME_MUTE:	nablaKeyCode = EKC_VOLUME_MUTE; break;
		case VK_VOLUME_UP:		nablaKeyCode = EKC_VOLUME_UP; break;
		case VK_VOLUME_DOWN:	nablaKeyCode = EKC_VOLUME_DOWN; break;
		case VK_F1:				nablaKeyCode = EKC_F1; break;
		case VK_F2:				nablaKeyCode = EKC_F2; break;
		case VK_F3:				nablaKeyCode = EKC_F3; break;
		case VK_F4:				nablaKeyCode = EKC_F4; break;
		case VK_F5:				nablaKeyCode = EKC_F5; break;
		case VK_F6:				nablaKeyCode = EKC_F6; break;
		case VK_F7:				nablaKeyCode = EKC_F7; break;
		case VK_F8:				nablaKeyCode = EKC_F8; break;
		case VK_F9:				nablaKeyCode = EKC_F9; break;
		case VK_F10:			nablaKeyCode = EKC_F10; break;
		case VK_F11:			nablaKeyCode = EKC_F11; break;
		case VK_F12:			nablaKeyCode = EKC_F12; break;
		case VK_F13:			nablaKeyCode = EKC_F13; break;
		case VK_F14:			nablaKeyCode = EKC_F14; break;
		case VK_F15:			nablaKeyCode = EKC_F15; break;
		case VK_F16:			nablaKeyCode = EKC_F16; break;
		case VK_F17:			nablaKeyCode = EKC_F17; break;
		case VK_F18:			nablaKeyCode = EKC_F18; break;
		case VK_F19:			nablaKeyCode = EKC_F19; break;
		case VK_F20:			nablaKeyCode = EKC_F20; break;
		case VK_F21:			nablaKeyCode = EKC_F21; break;
		case VK_F22:			nablaKeyCode = EKC_F22; break;
		case VK_F23:			nablaKeyCode = EKC_F23; break;
		case VK_F24:			nablaKeyCode = EKC_F24; break;

		
		}
		return nablaKeyCode;
	}

}
}

#endif