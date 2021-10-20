#ifndef _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#define _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CApplicationAndroid.h"
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"

#include "nbl/ui/IWindow.h"

namespace nbl::ui
{
	class CGraphicalApplicationAndroid : public system::CApplicationAndroid
	{
	public:
		struct SGraphicalContext : SContext
		{
			core::smart_refctd_ptr<nbl::system::ISystem> system;
			core::smart_refctd_ptr<CWindowManagerAndroid> wndManager;
			core::smart_refctd_ptr<IWindow::IEventCallback> callback;
		};
		CGraphicalApplicationAndroid(android_app* app, const system::path& cwd) : system::CApplicationAndroid(app, cwd) {}
	private:
		void handleInput_impl(android_app* data, AInputEvent* event) override
		{
			auto* wnd = getWindow();
			if (wnd != nullptr)
			{
				if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_KEY)
				{
					int32_t keyVal = AKeyEvent_getKeyCode(event);
					uint32_t deviceId = AInputEvent_getDeviceId(event);
					E_KEY_CODE kc = getNablaKeyCodeFromNative(keyVal);
					uint64_t eventTime = AKeyEvent_getEventTime(event);
					int32_t nativeAction = AKeyEvent_getAction(event);

					SKeyboardEvent::E_KEY_ACTION nblAction;
					switch (nativeAction)
					{
					case AKEY_EVENT_ACTION_DOWN:
						nblAction = SKeyboardEvent::ECA_PRESSED;
						break;
					case AKEY_EVENT_ACTION_UP:
						nblAction = SKeyboardEvent::ECA_RELEASED;
						break;
					}
					std::chrono::microseconds nblTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::nanoseconds(eventTime));
					SKeyboardEvent kbEvent(nblTime);
					kbEvent.action = nblAction;
					kbEvent.keyCode = kc;
					kbEvent.window = wnd;
				}
				else if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION)
				{
					uint32_t deviceId = AInputEvent_getDeviceId(event);
					int32_t rawAction = AMotionEvent_getAction(event);
					uint64_t eventTime = AMotionEvent_getEventTime(event);
					int32_t action = AMotionEvent_getAction(event);
					std::chrono::microseconds nblTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::nanoseconds(eventTime));
					SMouseEvent mouseEvent(nblTime);
					mouseEvent.window = wnd;
					if (action == AMOTION_EVENT_ACTION_HOVER_MOVE || action == AMOTION_EVENT_ACTION_MOVE)
					{
						mouseEvent.type = SMouseEvent::EET_MOVEMENT;
						mouseEvent.movementEvent.relativeMovementX = AMotionEvent_getHistoricalX(event, 0, 0);
						mouseEvent.movementEvent.relativeMovementY = AMotionEvent_getHistoricalY(event, 0, 0);
					}
					else if (action == AMOTION_EVENT_ACTION_BUTTON_PRESS)
					{
						mouseEvent.type = SMouseEvent::EET_CLICK;
						mouseEvent.clickEvent.clickPosX = AMotionEvent_getX(event, 0);
						mouseEvent.clickEvent.clickPosY = AMotionEvent_getY(event, 0);
						mouseEvent.clickEvent.action = SMouseEvent::SClickEvent::EA_PRESSED;
					}
					else if (action == AMOTION_EVENT_ACTION_BUTTON_RELEASE)
					{
						mouseEvent.type = SMouseEvent::EET_CLICK;
						mouseEvent.clickEvent.clickPosX = AMotionEvent_getX(event, 0);
						mouseEvent.clickEvent.clickPosY = AMotionEvent_getY(event, 0);
						mouseEvent.clickEvent.action = SMouseEvent::SClickEvent::EA_RELEASED;
					}
					else if (action == AMOTION_EVENT_ACTION_SCROLL)
					{
						mouseEvent.type = SMouseEvent::EET_SCROLL;
						mouseEvent.scrollEvent.verticalScroll = AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_VSCROLL, 0);
						mouseEvent.scrollEvent.horizontalScroll = AMotionEvent_getAxisValue(event, AMOTION_EVENT_AXIS_HSCROLL, 0);
					}


				}
			}
		}
		void handleCommand_impl(android_app* app, int32_t cmd) override
		{
			auto* ctx = (SGraphicalContext*)app->userData;
			switch (cmd)
			{
			case APP_CMD_INIT_WINDOW:
			{
				IWindow::SCreationParams params;
				params.callback = core::smart_refctd_ptr(ctx->callback);
				setWindow(ctx->wndManager->createWindow(std::move(params)));
				break;
			}
			case APP_CMD_TERM_WINDOW:
			{
				auto* wnd = getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onWindowClosed(wnd);
				}
				break;
			}
			case APP_CMD_WINDOW_RESIZED:
			{
				uint32_t width = ANativeWindow_getWidth(app->window);
				uint32_t height = ANativeWindow_getHeight(app->window);
				auto* wnd = getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onWindowResized(wnd, width, height);
				}

				break;
			}
			case APP_CMD_LOST_FOCUS:
			{
				auto* wnd = getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onLostMouseFocus(wnd);
					(void)eventCallback->onLostKeyboardFocus(wnd);
				}
				break;
			}
			case APP_CMD_GAINED_FOCUS:
			{
				auto* wnd = getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onGainedMouseFocus(wnd);
					(void)eventCallback->onGainedKeyboardFocus(wnd);
				}
				break;
			}
			}
		}
			static E_KEY_CODE getNablaKeyCodeFromNative(int32_t nativeKeyCode)
			{
				nbl::ui::E_KEY_CODE nablaKeyCode = EKC_NONE;
				switch (nativeKeyCode)
				{
				case AKEYCODE_BACK:				nablaKeyCode = EKC_BACKSPACE; break;
				case AKEYCODE_TAB:				nablaKeyCode = EKC_TAB; break;
				case AKEYCODE_CLEAR:			nablaKeyCode = EKC_CLEAR; break;
				case AKEYCODE_ENTER:			nablaKeyCode = EKC_ENTER; break;
				case AKEYCODE_SHIFT_RIGHT:		nablaKeyCode = EKC_RIGHT_SHIFT; break;
				case AKEYCODE_SHIFT_LEFT:		nablaKeyCode = EKC_LEFT_SHIFT; break;
				case AKEYCODE_CTRL_LEFT:		nablaKeyCode = EKC_LEFT_CONTROL; break;
				case AKEYCODE_CTRL_RIGHT:		nablaKeyCode = EKC_RIGHT_CONTROL; break;
				case AKEYCODE_ALT_LEFT:			nablaKeyCode = EKC_LEFT_ALT; break;
				case AKEYCODE_ALT_RIGHT:		nablaKeyCode = EKC_RIGHT_ALT; break;
				case AKEYCODE_MEDIA_PAUSE:		nablaKeyCode = EKC_PAUSE; break;
				case AKEYCODE_CAPS_LOCK:		nablaKeyCode = EKC_CAPS_LOCK; break;
				case AKEYCODE_ESCAPE:			nablaKeyCode = EKC_ESCAPE; break;
				case AKEYCODE_SPACE:			nablaKeyCode = EKC_SPACE; break;
				case AKEYCODE_PAGE_UP:			nablaKeyCode = EKC_PAGE_UP; break;
				case AKEYCODE_PAGE_DOWN:		nablaKeyCode = EKC_PAGE_DOWN; break;
				case AKEYCODE_MOVE_END:			nablaKeyCode = EKC_END; break;
				case AKEYCODE_HOME:				nablaKeyCode = EKC_HOME; break;
				case AKEYCODE_DPAD_LEFT:		nablaKeyCode = EKC_LEFT_ARROW; break;
				case AKEYCODE_DPAD_RIGHT:		nablaKeyCode = EKC_RIGHT_ARROW; break;
				case AKEYCODE_DPAD_UP:			nablaKeyCode = EKC_UP_ARROW; break;
				case AKEYCODE_DPAD_DOWN:		nablaKeyCode = EKC_DOWN_ARROW; break;
				case AKEYCODE_BUTTON_SELECT:	nablaKeyCode = EKC_SELECT; break;
				case AKEYCODE_SYSRQ:			nablaKeyCode = EKC_PRINT_SCREEN; break;
				case AKEYCODE_INSERT:			nablaKeyCode = EKC_INSERT; break;
				case AKEYCODE_DEL:				nablaKeyCode = EKC_DELETE; break;
				case AKEYCODE_HELP:				nablaKeyCode = EKC_HELP; break;
				case AKEYCODE_0:				nablaKeyCode = EKC_0; break;
				case AKEYCODE_1:				nablaKeyCode = EKC_1; break;
				case AKEYCODE_2:				nablaKeyCode = EKC_2; break;
				case AKEYCODE_3:				nablaKeyCode = EKC_3; break;
				case AKEYCODE_4:				nablaKeyCode = EKC_4; break;
				case AKEYCODE_5:				nablaKeyCode = EKC_5; break;
				case AKEYCODE_6:				nablaKeyCode = EKC_6; break;
				case AKEYCODE_7:				nablaKeyCode = EKC_7; break;
				case AKEYCODE_8:				nablaKeyCode = EKC_8; break;
				case AKEYCODE_9:				nablaKeyCode = EKC_9; break;
				case AKEYCODE_NUMPAD_0:		nablaKeyCode = EKC_NUMPAD_0; break;
				case AKEYCODE_NUMPAD_1:		nablaKeyCode = EKC_NUMPAD_1; break;
				case AKEYCODE_NUMPAD_2:		nablaKeyCode = EKC_NUMPAD_2; break;
				case AKEYCODE_NUMPAD_3:		nablaKeyCode = EKC_NUMPAD_3; break;
				case AKEYCODE_NUMPAD_4:		nablaKeyCode = EKC_NUMPAD_4; break;
				case AKEYCODE_NUMPAD_5:		nablaKeyCode = EKC_NUMPAD_5; break;
				case AKEYCODE_NUMPAD_6:		nablaKeyCode = EKC_NUMPAD_6; break;
				case AKEYCODE_NUMPAD_7:		nablaKeyCode = EKC_NUMPAD_7; break;
				case AKEYCODE_NUMPAD_8:		nablaKeyCode = EKC_NUMPAD_8; break;
				case AKEYCODE_NUMPAD_9:		nablaKeyCode = EKC_NUMPAD_9; break;
				case AKEYCODE_A:				nablaKeyCode = EKC_A; break;
				case AKEYCODE_B:				nablaKeyCode = EKC_B; break;
				case AKEYCODE_C:				nablaKeyCode = EKC_C; break;
				case AKEYCODE_D:				nablaKeyCode = EKC_D; break;
				case AKEYCODE_E:				nablaKeyCode = EKC_E; break;
				case AKEYCODE_F:				nablaKeyCode = EKC_F; break;
				case AKEYCODE_G:				nablaKeyCode = EKC_G; break;
				case AKEYCODE_H:				nablaKeyCode = EKC_H; break;
				case AKEYCODE_I:				nablaKeyCode = EKC_I; break;
				case AKEYCODE_J:				nablaKeyCode = EKC_J; break;
				case AKEYCODE_K:				nablaKeyCode = EKC_K; break;
				case AKEYCODE_L:				nablaKeyCode = EKC_L; break;
				case AKEYCODE_M:				nablaKeyCode = EKC_M; break;
				case AKEYCODE_N:				nablaKeyCode = EKC_N; break;
				case AKEYCODE_O:				nablaKeyCode = EKC_O; break;
				case AKEYCODE_P:				nablaKeyCode = EKC_P; break;
				case AKEYCODE_Q:				nablaKeyCode = EKC_Q; break;
				case AKEYCODE_R:				nablaKeyCode = EKC_R; break;
				case AKEYCODE_S:				nablaKeyCode = EKC_S; break;
				case AKEYCODE_T:				nablaKeyCode = EKC_T; break;
				case AKEYCODE_U:				nablaKeyCode = EKC_U; break;
				case AKEYCODE_V:				nablaKeyCode = EKC_V; break;
				case AKEYCODE_W:				nablaKeyCode = EKC_W; break;
				case AKEYCODE_X:				nablaKeyCode = EKC_X; break;
				case AKEYCODE_Y:				nablaKeyCode = EKC_Y; break;
				case AKEYCODE_Z:				nablaKeyCode = EKC_Z; break;
				case AKEYCODE_PLUS:				nablaKeyCode = EKC_ADD; break;
				case AKEYCODE_MINUS:			nablaKeyCode = EKC_SUBTRACT; break;
				case AKEYCODE_STAR:				nablaKeyCode = EKC_MULTIPLY; break;
				case AKEYCODE_SLASH:			nablaKeyCode = EKC_DIVIDE; break;
				case AKEYCODE_PERIOD: [[fallthrough]];
				case AKEYCODE_COMMA:			nablaKeyCode = EKC_SEPARATOR; break;
				case AKEYCODE_NUM_LOCK:			nablaKeyCode = EKC_NUM_LOCK; break;
				case AKEYCODE_SCROLL_LOCK:		nablaKeyCode = EKC_SCROLL_LOCK; break;
				case AKEYCODE_MUTE:				nablaKeyCode = EKC_VOLUME_MUTE; break;
				case AKEYCODE_VOLUME_UP:		nablaKeyCode = EKC_VOLUME_UP; break;
				case AKEYCODE_VOLUME_DOWN:		nablaKeyCode = EKC_VOLUME_DOWN; break;
				case AKEYCODE_F1:				nablaKeyCode = EKC_F1; break;
				case AKEYCODE_F2:				nablaKeyCode = EKC_F2; break;
				case AKEYCODE_F3:				nablaKeyCode = EKC_F3; break;
				case AKEYCODE_F4:				nablaKeyCode = EKC_F4; break;
				case AKEYCODE_F5:				nablaKeyCode = EKC_F5; break;
				case AKEYCODE_F6:				nablaKeyCode = EKC_F6; break;
				case AKEYCODE_F7:				nablaKeyCode = EKC_F7; break;
				case AKEYCODE_F8:				nablaKeyCode = EKC_F8; break;
				case AKEYCODE_F9:				nablaKeyCode = EKC_F9; break;
				case AKEYCODE_F10:				nablaKeyCode = EKC_F10; break;
				case AKEYCODE_F11:				nablaKeyCode = EKC_F11; break;
				case AKEYCODE_F12:				nablaKeyCode = EKC_F12; break;
				}
				return nablaKeyCode;
			}
			core::map<uint32_t, core::smart_refctd_ptr<IMouseEventChannel>> m_mouseEventChannels;
			core::map<uint32_t, core::smart_refctd_ptr<IKeyboardEventChannel>> m_keyboardEventChannels;
			bool addMouseEventChannel(uint32_t deviceId, const core::smart_refctd_ptr<IMouseEventChannel>&channel)
			{
				if (m_mouseEventChannels.find(deviceId) == m_mouseEventChannels.end())
				{
					m_mouseEventChannels.emplace(deviceId, channel);
					return true;
				}
				return false;
			}
			bool addKeyboardEventChannel(uint32_t deviceId, const core::smart_refctd_ptr<IKeyboardEventChannel>&channel)
			{
				if (m_keyboardEventChannels.find(deviceId) == m_keyboardEventChannels.end())
				{
					m_keyboardEventChannels.emplace(deviceId, channel);
					return true;
				}
				return false;
			}
			IMouseEventChannel* getMouseEventChannel(uint32_t deviceId)
			{
				auto ch = m_mouseEventChannels.find(deviceId);
				return m_mouseEventChannels.find(deviceId)->second.get();
			}

			IKeyboardEventChannel* getKeyboardEventChannel(uint32_t deviceId)
			{
				auto ch = m_keyboardEventChannels.find(deviceId);
				return m_keyboardEventChannels.find(deviceId)->second.get();
			}
		public:
			template<typename android_app_class, typename window_event_callback, typename ... EventCallbackArgs>
			static void androidMain(android_app * app, EventCallbackArgs... args)
			{
				system::path CWD = std::filesystem::current_path().generic_string();
				nbl::ui::CGraphicalApplicationAndroid::SGraphicalContext ctx{};
				app->userData = &ctx;
				auto framework = nbl::core::make_smart_refctd_ptr<android_app_class>(app, CWD);
				auto eventCallback = nbl::core::make_smart_refctd_ptr<window_event_callback>(std::forward<EventCallbackArgs>(args)...);
				auto wndManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerAndroid>(app);
				ctx.wndManager = core::smart_refctd_ptr(wndManager);
				ctx.callback = core::smart_refctd_ptr(eventCallback);
				nbl::ui::IWindow::SCreationParams params;
				params.callback = nullptr;
				auto system = core::make_smart_refctd_ptr<nbl::system::CSystemAndroid>(core::make_smart_refctd_ptr<nbl::system::CSystemCallerPOSIX>(), app->activity);
				framework->setSystem(std::move(system));
				if (app->savedState != nullptr) {
					ctx.state = *(nbl::system::CApplicationAndroid::SSavedState*)app->savedState;
				}
				android_poll_source* source;
				int ident;
				int events;
				while (framework->keepRunning()) {
					while ((ident = ALooper_pollAll(0, nullptr, &events, (void**)&source)) >= 0)
					{
						if (source != nullptr) {
							source->process(app, source);
						}
						if (app->destroyRequested != 0) {
							//todo
							return;
						}
					}
					if (app->window != nullptr && framework->getWindow() != nullptr)
						framework->workLoopBody();
				}
			}
	};
}

// ... are the window event callback optional ctor params;
#define NBL_ANDROID_MAIN_FUNC(android_app_class, window_event_callback, ...) void android_main(android_app* app){\
		nbl::ui::CGraphicalApplicationAndroid::androidMain<android_app_class, window_event_callback>(app __VA_ARGS__);\
    }

#endif
#endif