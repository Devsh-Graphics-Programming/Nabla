#ifndef _NBL_UI_C_WINDOWMANAGER_WIN32_
#define _NBL_UI_C_WINDOWMANAGER_WIN32_

#include "nbl/ui/IWindowManager.h"

#include <cstdint>
#include <queue>

#include "nbl/ui/CWindowWin32.h"

#ifdef _NBL_PLATFORM_WINDOWS_
#include <codecvt>
#include <xlocbuf>

//#include <hidpi.h>
#include <hidusage.h>

namespace nbl::ui
{

class CWindowManagerWin32 : public IWindowManager
{
	public:
		inline CWindowManagerWin32() = default;
		inline ~CWindowManagerWin32() {};

		inline core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final
		{
			CWindowWin32::native_handle_t handle = createNativeWindow(
				creationParams.x,
				creationParams.y,
				creationParams.width,
				creationParams.height,
				creationParams.flags,
				creationParams.windowCaption
			);

			if (handle == nullptr)
				return nullptr;

			return core::make_smart_refctd_ptr<CWindowWin32>(core::smart_refctd_ptr<CWindowManagerWin32>(this), std::move(creationParams), handle);
		}
		inline void destroyWindow(IWindow* wnd) override final
		{
			destroyNativeWindow(static_cast<IWindowWin32*>(wnd)->getNativeHandle());
		}
		inline void setCursorVisibility(bool visible)
		{
			m_windowThreadManager.setCursorVisibility(visible);
		}
		inline SDisplayInfo getPrimaryDisplayInfo() const override final
		{
			RECT size;
			BOOL res_ok = SystemParametersInfo(SPI_GETWORKAREA, 0, &size, 0);
			SDisplayInfo info {};
			info.resX = size.right - size.left;
			info.resY = size.bottom - size.top;
			info.x = size.left; // When would this be not 0 though??
			info.y = size.top;
			return info;
		}

		inline bool setWindowSize_impl(IWindow* window, const uint32_t width, const uint32_t height) override
		{
			// Calculate real window size based on client size
			RECT clientSize;
			clientSize.left = 0;
			clientSize.top = 0;
			clientSize.right = width;
			clientSize.bottom = height;

			DWORD style = IWindowWin32::getWindowStyle(window->getFlags().value);
			bool res = AdjustWindowRect(&clientSize, style, false);
			assert(res);

			const int32_t realWidth = clientSize.right - clientSize.left;
			const int32_t realHeight = clientSize.bottom - clientSize.top;

			auto wnd = static_cast<IWindowWin32*>(window);
			m_windowThreadManager.setWindowSize(wnd->getNativeHandle(), realWidth, realHeight);
			return true;
		}
		inline bool setWindowPosition_impl(IWindow* window, const int32_t x, const int32_t y) override
		{
			auto wnd = static_cast<IWindowWin32*>(window);
			m_windowThreadManager.setWindowPosition(wnd->getNativeHandle(), x, y);
			return true;
		}
		inline bool setWindowRotation_impl(IWindow* window, const bool landscape) override
		{
			return false;
		}
		inline bool setWindowVisible_impl(IWindow* window, const bool visible) override
		{
			auto wnd = static_cast<IWindowWin32*>(window);
			if (visible) m_windowThreadManager.showWindow(wnd->getNativeHandle());
			else m_windowThreadManager.hideWindow(wnd->getNativeHandle());
			return true;
		}
		inline bool setWindowMaximized_impl(IWindow* window, const bool maximized) override
		{
			auto wnd = static_cast<IWindowWin32*>(window);
			if (maximized) m_windowThreadManager.maximizeWindow(wnd->getNativeHandle());
			else m_windowThreadManager.minimizeWindow(wnd->getNativeHandle());
			return true;
		}

	private:
		inline IWindowWin32::native_handle_t createNativeWindow(int _x, int _y, uint32_t _w, uint32_t _h, IWindow::E_CREATE_FLAGS _flags, const std::string_view& caption)
		{
			IWindowWin32::native_handle_t out_handle;
			m_windowThreadManager.createWindow(_x, _y, _w, _h, _flags, &out_handle, caption);
			return out_handle;
		}
		inline void destroyNativeWindow(IWindowWin32::native_handle_t wnd)
		{
			m_windowThreadManager.destroyWindow(wnd);
		}

	private:
		enum E_REQUEST_TYPE
		{
			ERT_CREATE_WINDOW,
			ERT_DESTROY_WINDOW,
			ERT_CHANGE_CURSOR_VISIBILITY,
			ERT_SET_WINDOW_POS,
			ERT_SHOW_WINDOW
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
		struct SRequestParams_ChangeCursorVisibility : SRequestParamsBase<ERT_CHANGE_CURSOR_VISIBILITY>
		{
			bool visible;
		};
		struct SRequestParams_SetWindowPos : SRequestParamsBase<ERT_SET_WINDOW_POS>
		{
			IWindowWin32::native_handle_t window;
			int x, y;
			uint32_t width, height;

			uint32_t ignoreXY : 1 = 0;
			uint32_t ignoreSize : 1 = 0;
		};
		struct SRequestParams_ShowWindow : SRequestParamsBase<ERT_SHOW_WINDOW>
		{
			IWindowWin32::native_handle_t window;
			uint32_t hide : 1 = 0;
			uint32_t show : 1 = 0;
			uint32_t minimized : 1 = 0;
			uint32_t maximized : 1 = 0;
		};
		struct SRequest : system::impl::IAsyncQueueDispatcherBase::request_base_t
		{
			E_REQUEST_TYPE type;
			union
			{
				SRequestParams_CreateWindow createWindowParam;
				SRequestParams_DestroyWindow destroyWindowParam;
				SRequestParams_ChangeCursorVisibility changeCursorVisibilityParam;
				SRequestParams_SetWindowPos setWindowPosParam;
				SRequestParams_ShowWindow showWindowParam;
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
				inline void createWindow(int32_t _x, int32_t _y, uint32_t _w, uint32_t _h, CWindowWin32::E_CREATE_FLAGS _flags, CWindowWin32::native_handle_t* wnd, const std::string_view& caption)
				{
					SRequestParams_CreateWindow params = SRequestParams_CreateWindow(_x, _y, _w, _h, _flags, wnd, caption);
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void destroyWindow(CWindowWin32::native_handle_t window)
				{
					SRequestParams_DestroyWindow params;
					params.nativeWindow = window;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void setCursorVisibility(bool visible)
				{
					SRequestParams_ChangeCursorVisibility params;
					params.visible = visible;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void setWindowSize(IWindowWin32::native_handle_t window, uint32_t width, uint32_t height)
				{
					SRequestParams_SetWindowPos params;
					params.window = window;
					params.width = width;
					params.height = height;
					params.ignoreXY = 1;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void setWindowPosition(IWindowWin32::native_handle_t window, int x, int y)
				{
					SRequestParams_SetWindowPos params;
					params.window = window;
					params.x = x;
					params.y = y;
					params.ignoreSize = 1;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void hideWindow(IWindowWin32::native_handle_t window)
				{
					SRequestParams_ShowWindow params;
					params.window = window;
					params.hide = 1;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void showWindow(IWindowWin32::native_handle_t window)
				{
					SRequestParams_ShowWindow params;
					params.window = window;
					params.show = 1;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void maximizeWindow(IWindowWin32::native_handle_t window)
				{
					SRequestParams_ShowWindow params;
					params.window = window;
					params.maximized = 1;
					auto& rq = request(params);
					waitForCompletion(rq);
				}
				inline void minimizeWindow(IWindowWin32::native_handle_t window)
				{
					SRequestParams_ShowWindow params;
					params.window = window;
					params.minimized = 1;
					auto& rq = request(params);
					waitForCompletion(rq);
				}

				inline CThreadHandler()
				{
					this->start();
				}
				inline ~CThreadHandler()
				{
				}

			private:
				void waitForCompletion(SRequest& req)
				{
					req.wait_ready();
					req.discard_storage();
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

						const char* classname = "Nabla Engine";

						WNDCLASSEXA wcex;
						wcex.cbSize = sizeof(WNDCLASSEX);
						wcex.style = CS_HREDRAW | CS_VREDRAW;
						wcex.lpfnWndProc = CWindowWin32::WndProc;
						wcex.cbClsExtra = 0;
						wcex.cbWndExtra = 0;
						wcex.hInstance = hinstance;
						wcex.hIcon = NULL;
						wcex.hCursor = nullptr;
						wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
						wcex.lpszMenuName = 0;
						wcex.lpszClassName = classname;
						wcex.hIconSm = 0;

						RegisterClassExA(&wcex);
						// calculate client size

						RECT clientSize;
						clientSize.top = params.y;
						clientSize.left = params.x;
						clientSize.right = clientSize.left + params.width;
						clientSize.bottom = clientSize.top + params.height;

						DWORD style = IWindowWin32::getWindowStyle(params.flags);

						// TODO:
						// if (hasMouseCaptured())
						// if (hasInputFocus())
						// if (hasMouseFocus())

						AdjustWindowRect(&clientSize, style, FALSE);

						const int32_t realWidth = clientSize.right - clientSize.left;
						const int32_t realHeight = clientSize.bottom - clientSize.top;

					
						*params.nativeWindow = CreateWindowA(classname, params.windowCaption.c_str(), style, clientSize.left, clientSize.top,
							realWidth, realHeight, NULL, NULL, hinstance, NULL);
						if ((params.flags & CWindowWin32::ECF_HIDDEN) == 0)
							ShowWindow(*params.nativeWindow, SW_SHOWNORMAL);
						UpdateWindow(*params.nativeWindow);

						// fix ugly ATI driver bugs. Thanks to ariaci
						// TODO still needed?
						MoveWindow(*params.nativeWindow, clientSize.left, clientSize.top, realWidth, realHeight, TRUE);
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
					case ERT_CHANGE_CURSOR_VISIBILITY:
					{
						auto& params = req.changeCursorVisibilityParam;
						if(params.visible)
						{
							int ret = ShowCursor(true);
							while (ret < 0)  ret = ShowCursor(true);
						}
						else
						{
							int ret = ShowCursor(false);
							while (ret >= 0)  ret = ShowCursor(false);
						}
						break;
					}
					case ERT_SET_WINDOW_POS:
					{
						auto& params = req.setWindowPosParam;
						uint32_t flags = SWP_NOACTIVATE | SWP_NOZORDER;
						if (params.ignoreXY) flags |= SWP_NOMOVE | SWP_NOREPOSITION;
						if (params.ignoreSize) flags |= SWP_NOSIZE;
						SetWindowPos(params.window, nullptr, params.x, params.y, params.width, params.height, flags);
						break;
					}
					case ERT_SHOW_WINDOW:
					{
						auto& params = req.showWindowParam;
						int showCmd;
						if (params.hide) showCmd = SW_HIDE;
						if (params.show) showCmd = SW_SHOW;
						if (params.minimized) showCmd = SW_MINIMIZE;
						if (params.maximized) showCmd = SW_MAXIMIZE;
						ShowWindow(params.window, showCmd);
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
					else if constexpr (std::is_same_v<RequestParams, SRequestParams_ChangeCursorVisibility&>)
					{
						req.changeCursorVisibilityParam = std::move(params);
					}
					else if constexpr (std::is_same_v<RequestParams, SRequestParams_SetWindowPos&>)
					{
						req.setWindowPosParam = std::move(params);
					}
					else if constexpr (std::is_same_v<RequestParams, SRequestParams_ShowWindow&>)
					{
						req.showWindowParam = std::move(params);
					}
				}

				inline bool wakeupPredicate() const { return true; }
				inline bool continuePredicate() const { return true; }

			private:
				static inline bool getMessageWithTimeout(MSG* msg, uint32_t timeoutInMilliseconds)
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