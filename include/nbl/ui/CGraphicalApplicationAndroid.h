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
		void handleCommand_impl(android_app* app, int32_t cmd) override
		{
			auto* ctx = (SGraphicalContext*)app->userData;
			auto* usrData = ((IUserData*)((SGraphicalContext*)app->userData)->userData);
			switch (cmd)
			{
			case APP_CMD_INIT_WINDOW:
			{
				IWindow::SCreationParams params;
				params.callback = core::smart_refctd_ptr(ctx->callback);
				usrData->setWindow(ctx->wndManager->createWindow(std::move(params)));
				break;
			}
			case APP_CMD_TERM_WINDOW:
			{
				auto* wnd = usrData->getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onWindowClosed(wnd);
				}
				break;
			}
			case APP_CMD_WINDOW_RESIZED:
			{
				int width = ANativeWindow_getWidth(app->window);
				int height = ANativeWindow_getHeight(app->window);
				SGraphicalContext* usrdata = (SGraphicalContext*)app->userData;
				auto* wnd = usrData->getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onWindowResized(wnd, width, height);
				}
				
				break;
			}
			case APP_CMD_LOST_FOCUS:
			{
				auto* wnd = usrData->getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onGainedMouseFocus(wnd);
					(void)eventCallback->onGainedKeyboardFocus(wnd);
				}
				break;
			}
			case APP_CMD_GAINED_FOCUS:
			{
				auto* wnd = usrData->getWindow();
				if (wnd != nullptr)
				{
					auto eventCallback = wnd->getEventCallback();
					(void)eventCallback->onLostMouseFocus(wnd);
					(void)eventCallback->onLostKeyboardFocus(wnd);
				}
				break;
			}
			}
		}
	public:
		template<typename android_app_class, typename user_data_type, typename window_event_callback, typename ... EventCallbackArgs>
		static void androidMain(android_app* app, EventCallbackArgs... args)
		{
			static_assert(std::is_base_of_v<nbl::system::IApplicationFramework::IUserData, user_data_type>);
			system::path CWD = std::filesystem::current_path().generic_string();
			user_data_type engine{};
			nbl::ui::CGraphicalApplicationAndroid::SGraphicalContext ctx{};
			ctx.userData = &engine;
			app->userData = &ctx;
			auto framework = nbl::core::make_smart_refctd_ptr<android_app_class>(app, CWD);
			auto eventCallback = nbl::core::make_smart_refctd_ptr<window_event_callback>(std::forward<EventCallbackArgs>(args)...);
			auto wndManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerAndroid>(app);
			ctx.wndManager = core::smart_refctd_ptr(wndManager);
			ctx.callback = core::smart_refctd_ptr(eventCallback);
			nbl::ui::IWindow::SCreationParams params;
			params.callback = nullptr;
			auto system = core::make_smart_refctd_ptr<nbl::system::CSystemAndroid>(core::make_smart_refctd_ptr<nbl::system::CSystemCallerPOSIX>(), app->activity);
			engine.setSystem(std::move(system));
			if (app->savedState != nullptr) {
				ctx.state = *(nbl::system::CApplicationAndroid::SSavedState*)app->savedState;
			}
			android_poll_source* source;
			int ident;
			int events;
			while (framework->keepRunning(&engine)) {
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
				if (app->window != nullptr  && engine.getWindow() != nullptr)
					framework->workLoopBody(&engine); 
			}
		}
	};
}

// ... are the window event callback optional ctor params;
#define NBL_ANDROID_MAIN_FUNC(android_app_class, user_data_type, window_event_callback, ...) void android_main(android_app* app){\
		nbl::ui::CGraphicalApplicationAndroid::androidMain<android_app_class, user_data_type, window_event_callback>(app __VA_ARGS__);\
    }

#endif
#endif