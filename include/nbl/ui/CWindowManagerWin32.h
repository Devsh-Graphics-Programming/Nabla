#ifndef C_WINDOWMANAGER_WIN32
#define C_WINDOWMANAGER_WIN32

#include "nbl/ui/IWindowManager.h"
#include <cstdint>
#include <queue>

#ifdef _NBL_PLATFORM_WINDOWS_
#include <codecvt>
#include <xlocbuf>

#include <Windows.h>
//#include <hidpi.h>
#include <hidusage.h>

#include "nbl/ui/CWindowWin32.h"

namespace nbl::ui
{
	class CWindowManagerWin32 : public IWindowManager
	{
	public:
		CWindowManagerWin32() = default;
		~CWindowManagerWin32() {};
		core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final
		{
			CWindowWin32::native_handle_t handle = createNativeWindow(creationParams.x,
				creationParams.y,
				creationParams.width,
				creationParams.height,
				creationParams.flags,
				creationParams.windowCaption);
			if (handle == nullptr)
			{
				return nullptr;
			}
			return core::make_smart_refctd_ptr<CWindowWin32>(core::smart_refctd_ptr<CWindowManagerWin32>(this), std::move(creationParams), handle);
		}
		void destroyWindow(IWindow* wnd) override final
		{
			destroyNativeWindow(static_cast<IWindowWin32*>(wnd)->getNativeHandle());
		}
	private:
		IWindowWin32::native_handle_t createNativeWindow(int _x, int _y, uint32_t _w, uint32_t _h, IWindow::E_CREATE_FLAGS _flags, const std::string_view& caption)
		{
			IWindowWin32::native_handle_t out_handle;
			m_windowThreadManager.createWindow(_x, _y, _w, _h, _flags, &out_handle, caption);
			return out_handle;
		}
		void destroyNativeWindow(IWindowWin32::native_handle_t wnd)
		{
			m_windowThreadManager.destroyWindow(wnd);
		}
	private:
		enum E_REQUEST_TYPE
		{
			ERT_CREATE_WINDOW,
			ERT_DESTROY_WINDOW
		};
		template <E_REQUEST_TYPE ERT>
		struct SRequestParamsBase
		{
			static inline constexpr E_REQUEST_TYPE type = ERT;
		};
		struct SRequestParams_CreateWindow : SRequestParamsBase<ERT_CREATE_WINDOW>
		{
			SRequestParams_CreateWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t* wnd, const std::string_view& caption) :
				x(_x), y(_y), width(_w), height(_h), flags(_flags), nativeWindow(wnd), windowCaption(caption)
			{
			}
			int32_t x, y;
			uint32_t width, height;
			CWindowWin32::E_CREATE_FLAGS flags;
			CWindowWin32::native_handle_t* nativeWindow;
			std::string windowCaption;
		};
		struct SRequestParams_DestroyWindow : SRequestParamsBase<ERT_DESTROY_WINDOW>
		{
			CWindowWin32::native_handle_t nativeWindow;
		};
		struct SRequest : system::impl::IAsyncQueueDispatcherBase::request_base_t
		{
			E_REQUEST_TYPE type;
			union
			{
				SRequestParams_CreateWindow createWindowParam;
				SRequestParams_DestroyWindow destroyWindowParam;
			};
			SRequest() {}
			~SRequest() {}
		};

		class CThreadHandler final : public system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u>
		{
			using base_t = system::IAsyncQueueDispatcher<CThreadHandler, SRequest, 256u>;
			friend base_t;
			friend base_t::base_t;
		public:
			void createWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t* wnd, const std::string_view& caption)
			{
				SRequestParams_CreateWindow params = SRequestParams_CreateWindow(_x, _y, _w, _h, _flags, wnd, caption);
				auto& rq = request(params);
				waitForCompletion(rq);
			}
			void destroyWindow(CWindowWin32::native_handle_t window)
			{
				SRequestParams_DestroyWindow params;
				params.nativeWindow = window;
				auto& rq = request(params);
				waitForCompletion(rq);
			}
			CThreadHandler()
			{
				this->start();
			}
			~CThreadHandler()
			{
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
				MSG message;
				constexpr uint32_t timeoutInMS = 8; // gonna become 10 anyway
				if (getMessageWithTimeout(&message, timeoutInMS))
				{
					TranslateMessage(&message);
					DispatchMessage(&message);
				}
			}

			void process_request(SRequest& req)
			{
				switch (req.type)
				{
				case ERT_CREATE_WINDOW:
				{
					auto& params = req.createWindowParam;
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
					wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
					wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
					wcex.lpszMenuName = 0;
					wcex.lpszClassName = classname;
					wcex.hIconSm = 0;

					RegisterClassEx(&wcex);

					// calculate client size

					RECT clientSize;
					clientSize.top = params.y;
					clientSize.left = params.x;
					clientSize.right = clientSize.left + params.width;
					clientSize.bottom = clientSize.top + params.height;

					DWORD style = WS_POPUP; // TODO why popup?

					if ((params.flags & CWindowWin32::ECF_FULLSCREEN) == 0)
					{
						if ((params.flags & CWindowWin32::ECF_BORDERLESS) == 0)
						{
							style |= WS_BORDER;
							style |= (WS_SYSMENU | WS_CAPTION);
						}
						// ? not sure about those below
						style |= WS_CLIPCHILDREN;
						style |= WS_CLIPSIBLINGS;
					}
					if (params.flags & CWindowWin32::ECF_MINIMIZED)
					{
						style |= WS_MINIMIZE;
					}
					if (params.flags & CWindowWin32::ECF_MAXIMIZED)
					{
						style |= WS_MAXIMIZE;
					}
					if (params.flags & CWindowWin32::ECF_ALWAYS_ON_TOP)
					{
						style |= WS_EX_TOPMOST;
					}
					if ((params.flags & CWindowWin32::ECF_HIDDEN) == 0)
					{
						style |= WS_VISIBLE;
					}
					style |= WS_OVERLAPPEDWINDOW;

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

					if (params.flags & CWindowWin32::ECF_FULLSCREEN)
					{
						windowLeft = 0;
						windowTop = 0;
					}
					*params.nativeWindow = CreateWindow(classname, params.windowCaption.c_str(), style, windowLeft, windowTop,
						realWidth, realHeight, NULL, NULL, hinstance, NULL);
					if ((params.flags & CWindowWin32::ECF_HIDDEN) == 0)
						ShowWindow(*params.nativeWindow, SW_SHOWNORMAL);
					UpdateWindow(*params.nativeWindow);

					// fix ugly ATI driver bugs. Thanks to ariaci
					// TODO still needed?
					MoveWindow(*params.nativeWindow, windowLeft, windowTop, realWidth, realHeight, TRUE);
					{
						//TODO: thoroughly test this stuff	(what is this about, you need to register devices yourself!? I thought Windows can give you a list of raw input devices!?)
						constexpr uint32_t INPUT_DEVICES_COUNT = 5;
						RAWINPUTDEVICE inputDevices[INPUT_DEVICES_COUNT];
						inputDevices[0].hwndTarget = *params.nativeWindow;
						inputDevices[0].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
						inputDevices[0].usUsagePage = HID_USAGE_PAGE_GENERIC;
						inputDevices[0].usUsage = HID_USAGE_GENERIC_POINTER;

						inputDevices[1].hwndTarget = *params.nativeWindow;
						inputDevices[1].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
						inputDevices[1].usUsagePage = HID_USAGE_PAGE_GENERIC;
						inputDevices[1].usUsage = HID_USAGE_GENERIC_MOUSE;

						inputDevices[2].hwndTarget = *params.nativeWindow;
						inputDevices[2].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
						inputDevices[2].usUsagePage = HID_USAGE_PAGE_GENERIC;
						inputDevices[2].usUsage = HID_USAGE_GENERIC_KEYBOARD;

						inputDevices[3].hwndTarget = *params.nativeWindow;
						inputDevices[3].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
						inputDevices[3].usUsagePage = HID_USAGE_PAGE_GAME;
						inputDevices[3].usUsage = HID_USAGE_GENERIC_JOYSTICK;

						inputDevices[4].hwndTarget = *params.nativeWindow;
						inputDevices[4].dwFlags = RIDEV_DEVNOTIFY | RIDEV_INPUTSINK;
						inputDevices[4].usUsagePage = HID_USAGE_PAGE_GAME;
						inputDevices[4].usUsage = HID_USAGE_GENERIC_GAMEPAD;

						RegisterRawInputDevices(inputDevices, INPUT_DEVICES_COUNT, sizeof(RAWINPUTDEVICE));
					}
					break;
				}
				case ERT_DESTROY_WINDOW:
				{
					auto& params = req.destroyWindowParam;
					DestroyWindow(params.nativeWindow);
					break;
				}
				}
			}

			template <typename RequestParams>
			void request_impl(SRequest& req, RequestParams&& params)
			{
				req.type = params.type;
				if constexpr (std::is_same_v<RequestParams, SRequestParams_CreateWindow&>)
				{
					req.createWindowParam = std::move(params);
				}
				else if constexpr (std::is_same_v<RequestParams, SRequestParams_DestroyWindow&>)
				{
					req.destroyWindowParam = std::move(params);
				}
			}

			bool wakeupPredicate() const { return true; }
			bool continuePredicate() const { return true; }
		private:
			static bool getMessageWithTimeout(MSG* msg, uint32_t timeoutInMilliseconds)
			{
				bool res;
				UINT_PTR timerId = SetTimer(NULL, NULL, timeoutInMilliseconds, NULL);
				res = GetMessage(msg, nullptr, 0, 0);

				KillTimer(NULL, timerId);

				if (!res)
					return false;
				if (msg->message == WM_TIMER && msg->hwnd == NULL && msg->wParam == timerId)
					return false;
				return true;
			}
		} m_windowThreadManager;
	};
}
#endif
#endif