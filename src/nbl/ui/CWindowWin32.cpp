#include "nbl/ui/CWindowWin32.h"

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
		SCreateWindowRequest(int _x, int_y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t wnd) : 
			m_params{ _x, _y, _w, _h, _flags, wnd } {}
		struct SCreateWindowParams : SParams
		{
			int x, y;
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

			// create window

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
		void createWindow(int _x, int _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t wnd)
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
		case WM_MOVING: [[fallthrough]]
		case WM_MOVE:
		{
			int newX = (int)LOWORD(lParam);
			int newY = (int)HIWORD(lParam);
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