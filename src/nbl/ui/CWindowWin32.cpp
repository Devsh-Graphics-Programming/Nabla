#include "nbl/ui/CWindowWin32.h"
#include <hidusage.h>

#ifdef _NBL_PLATFORM_WINDOWS_

namespace nbl {
namespace ui
{
	
	struct SRequest : system::impl::IAsyncQueueDispatcherBase::request_base_t
	{
		struct SParams {};
		virtual void process() = 0;
		virtual void setParams(SParams&& params) = 0;
	};
	struct SCreateWindowRequest : SRequest
	{
		SCreateWindowRequest(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t wnd) :
			m_params{ _x, _y, _w, _h, _flags, wnd } {}
		struct SCreateWindowParams : SParams
		{
			int32_t x, y;
			uint32_t width, height;
			CWindowWin32::E_CREATE_FLAGS flags;
			CWindowWin32::native_handle_t nativeWindow;
		} m_params;
		void process() override
		{
			HINSTANCE hinstance = GetModuleHandle(NULL);

			const char* classname = __TEXT("Nabla Engine");

			WNDCLASSEX wcex;
			wcex.cbSize = sizeof(WNDCLASSEX);
			wcex.style = CS_HREDRAW | CS_VREDRAW;
			wcex.lpfnWndProc = CWindowWin32::WndProc;
			wcex.cbClsExtra = 0;
			wcex.cbWndExtra = 0;
			wcex.hInstance = hinstance;
			wcex.hIcon = NULL;
			wcex.hCursor = 0; // LoadCursor(NULL, IDC_ARROW);
			wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
			wcex.lpszMenuName = 0;
			wcex.lpszClassName = classname;
			wcex.hIconSm = 0;

			RegisterClassEx(&wcex);

			// calculate client size

			RECT clientSize;
			clientSize.top = m_params.y;
			clientSize.left = m_params.x;
			clientSize.right = clientSize.left + m_params.width;
			clientSize.bottom = clientSize.top + m_params.height;

			DWORD style = WS_POPUP; // TODO why popup?

			if ((m_params.flags & CWindowWin32::ECF_FULLSCREEN) == 0)
			{
				if ((m_params.flags & CWindowWin32::ECF_BORDERLESS) == 0)
				{
					style |= WS_BORDER;
					style |= (WS_SYSMENU | WS_CAPTION);
				}
				// ? not sure about those below
				style |= WS_CLIPCHILDREN;
				style |= WS_CLIPSIBLINGS;
			}
			if (m_params.flags & CWindowWin32::ECF_MINIMIZED)
			{
				style |= WS_MINIMIZE;
			}
			if (m_params.flags & CWindowWin32::ECF_MAXIMIZED)
			{
				style |= WS_MAXIMIZE;
			}
			if (m_params.flags & CWindowWin32::ECF_ALWAYS_ON_TOP)
			{
				style |= WS_EX_TOPMOST;
			}
			if ((m_params.flags & CWindowWin32::ECF_HIDDEN) == 0)
			{
				style |= WS_VISIBLE;
			}


			// TODO:
			// if (hasMouseCaptured())
			// if (hasInputFocus())
			// if (hasMouseFocus())

			AdjustWindowRect(&clientSize, style, FALSE);

			const int32_t realWidth = clientSize.right - clientSize.left;
			const int32_t realHeight = clientSize.bottom - clientSize.top;

			int32_t windowLeft = (GetSystemMetrics(SM_CXSCREEN) - realWidth) / 2;
			int32_t windowTop = (GetSystemMetrics(SM_CYSCREEN) - realHeight) / 2;

			if (windowLeft < 0)
				windowLeft = 0;
			if (windowTop < 0)
				windowTop = 0;	// make sure window menus are in screen on creation

			if (m_params.flags & CWindowWin32::ECF_FULLSCREEN)
			{
				windowLeft = 0;
				windowTop = 0;
			}
			{
				//TODO: thoroughly test this stuff	
				constexpr uint32_t INPUT_DEVICES_COUNT = 5;
				RAWINPUTDEVICE inputDevices[INPUT_DEVICES_COUNT];
				inputDevices[0].hwndTarget = m_params.nativeWindow;
				inputDevices[0].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
				inputDevices[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
				inputDevices[0].usUsage = HID_USAGE_GENERIC_POINTER;

				inputDevices[1].hwndTarget = m_params.nativeWindow;
				inputDevices[1].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
				inputDevices[1].usUsagePage = HID_USAGE_PAGE_GENERIC;
				inputDevices[1].usUsage = HID_USAGE_GENERIC_MOUSE;

				inputDevices[2].hwndTarget = m_params.nativeWindow;
				inputDevices[2].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
				inputDevices[2].usUsagePage = HID_USAGE_PAGE_GENERIC;
				inputDevices[2].usUsage = HID_USAGE_GENERIC_KEYBOARD;

				inputDevices[3].hwndTarget = m_params.nativeWindow;
				inputDevices[3].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
				inputDevices[3].usUsagePage = HID_USAGE_PAGE_GAME;
				inputDevices[3].usUsage = HID_USAGE_GENERIC_JOYSTICK;

				inputDevices[4].hwndTarget = m_params.nativeWindow;
				inputDevices[4].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
				inputDevices[4].usUsagePage = HID_USAGE_PAGE_GAME;
				inputDevices[4].usUsage = HID_USAGE_GENERIC_GAMEPAD;

				RegisterRawInputDevices(inputDevices, INPUT_DEVICES_COUNT, sizeof(RAWINPUTDEVICE));
			}
			m_params.nativeWindow = CreateWindow(classname, __TEXT(""), style, windowLeft, windowTop,
				realWidth, realHeight, NULL, NULL, hinstance, NULL);
			if ((m_params.flags & CWindowWin32::ECF_HIDDEN) == 0)
				ShowWindow(m_params.nativeWindow, SW_SHOWNORMAL);
			UpdateWindow(m_params.nativeWindow);

			// fix ugly ATI driver bugs. Thanks to ariaci
			// TODO still needed?
			MoveWindow(m_params.nativeWindow, windowLeft, windowTop, realWidth, realHeight, TRUE);
		}
		void setParams(SParams&& params) override { m_params = static_cast<SCreateWindowParams&&>(params); }
	};
	struct SDestroyWindowRequest : SRequest
	{
		SDestroyWindowRequest(CWindowWin32::native_handle_t wnd) : m_params{ wnd } {}
		struct SDestroyWindowParams : SParams
		{
			CWindowWin32::native_handle_t window;
		} m_params;
		void process() override
		{
			DestroyWindow(m_params.window);
		}
		void setParams(SParams&& params) override { m_params = static_cast<SDestroyWindowParams&&>(params); }
	};
	class CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u>
	{
		using base_t = system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u>;
		friend base_t;
	public:
		void createWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t wnd)
		{
			SCreateWindowRequest rq(_x, _y, _w, _h, _flags, wnd);
			request(rq);
			waitForCompletion(rq);
		}
		void destroyWindow(CWindowWin32::native_handle_t window)
		{
			SDestroyWindowRequest rq(window);
			request(rq);
			waitForCompletion(rq);
		}
		CThreadHandler()
		{
			this->start();
		}

	private:
		void waitForCompletion(SRequest& req)
		{
			auto lk = req.wait();
		}

	private:
		void init() {}

		void exit() {}

		void background_work()
		{
			static MSG message;
			static uint32_t timeoutInMS = 8; // gonna become 10 anyway
			if (getMessageWithTimeout(&message, timeoutInMS))
			{
				TranslateMessage(&message);
				DispatchMessage(&message);
			}
		}

		void process_request(SRequest& req)
		{
			req.process();
		}

		template <typename RequestParams>
		void request_impl(SRequest& req, RequestParams&& params)
		{
			req.setParams(std::move(params));
		}
	private:
		static bool getMessageWithTimeout(MSG* msg, uint32_t timeoutInMilliseconds)
		{
			bool res;
			UINT_PTR timerId = SetTimer(NULL, NULL, timeoutInMilliseconds, NULL);
			res = GetMessage(msg, nullptr, 0, 0);
			
			PostMessage(nullptr, WM_NULL, 0, 0);
			//KillTimer(NULL, timerId);

			if (!res)
				return false;
			if (msg->message == WM_TIMER && msg->hwnd == NULL && msg->wParam == timerId)
				return false;
			return true;
		}
	} windowThreadHandler; 

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
 
				if ((rawMouse.usFlags & MOUSE_MOVE_RELATIVE) == MOUSE_MOVE_RELATIVE && (rawMouse.lLastX != 0 || rawMouse.lLastY != 0))
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_MOTION;
					event.motionEvent.motionX = rawMouse.lLastX;
					event.motionEvent.motionY = rawMouse.lLastY;
					event.window = window;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				if (rawMouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_BUTTON;
					event.buttonEvent.state = SMouseEvent::SButtonEvent::ES_PRESSED;
					event.buttonEvent.button = ui::E_MOUSEBUTTON::EMB_LEFT_BUTTON;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_BUTTON;
					event.buttonEvent.state = SMouseEvent::SButtonEvent::ES_RELEASED;
					event.buttonEvent.button = ui::E_MOUSEBUTTON::EMB_LEFT_BUTTON;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				if (rawMouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_BUTTON;
					event.buttonEvent.state = SMouseEvent::SButtonEvent::ES_PRESSED;
					event.buttonEvent.button = ui::E_MOUSEBUTTON::EMB_RIGHT_BUTTON;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_BUTTON;
					event.buttonEvent.state = SMouseEvent::SButtonEvent::ES_RELEASED;
					event.buttonEvent.button = ui::E_MOUSEBUTTON::EMB_RIGHT_BUTTON;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				if (rawMouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_BUTTON;
					event.buttonEvent.state = SMouseEvent::SButtonEvent::ES_PRESSED;
					event.buttonEvent.button = ui::E_MOUSEBUTTON::EMB_MIDDLE_BUTTON;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				else if (rawMouse.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)
				{
					SMouseEvent event;
					event.type = SMouseEvent::EMT_BUTTON;
					event.buttonEvent.state = SMouseEvent::SButtonEvent::ES_RELEASED;
					event.buttonEvent.button = ui::E_MOUSEBUTTON::EMB_MIDDLE_BUTTON;
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				// TODO other mouse buttons

				if (rawMouse.usButtonFlags & RI_MOUSE_WHEEL)
				{
					SHORT wheelDelta = static_cast<SHORT>(rawMouse.usButtonData);
					SMouseEvent event;
					event.type = SMouseEvent::EMT_WHEEL;
					event.motionEvent.verticalDelta = wheelDelta;
					event.motionEvent.horizontalDelta = 0; // TODO horizontal wheel (lol never seen one)
					auto lk = inputChannel->lockBackgroundBuffer();
					lk.lock();
					inputChannel->pushIntoBackground(std::move(event));
				}
				break;
			}
			case RIM_TYPEKEYBOARD:
			{
				auto inputChannel = window->getKeyboardEventChannel(device);

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

}
}

#endif