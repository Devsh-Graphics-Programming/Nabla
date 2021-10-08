#ifndef _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#define _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CApplicationAndroid.h"
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"

namespace nbl::ui
{
	class CGraphicalApplicationAndroid : public system::CApplicationAndroid
	{
	public:
		struct SGraphicalContext : SContext
		{
			core::smart_refctd_ptr<nbl::ui::IWindow> window;
			core::smart_refctd_ptr<nbl::system::ISystem> system;
		};
		CGraphicalApplicationAndroid(android_app* app, const system::path& cwd) : system::CApplicationAndroid(app, cwd) {}
	private:
		void handleCommand_impl(android_app* app, int32_t cmd) override
		{
			switch (cmd)
			{
			case APP_CMD_INIT_WINDOW:
			{
				auto& wnd = ((SGraphicalContext*)app->userData)->window;
				((IUserData*)((SGraphicalContext*)app->userData)->userData)->setWindow(core::make_smart_refctd_ptr<nbl::ui::CWindowAndroid>(app->window));
				break;
			}
			case APP_CMD_TERM_WINDOW:
			{
				auto wnd = ((SGraphicalContext*)app->userData)->window;
				auto eventCallback = wnd->getEventCallback();
				(void)eventCallback->onWindowClosed(wnd.get());
				break;
			}
			case APP_CMD_WINDOW_RESIZED:
			{
				auto wnd = ((SGraphicalContext*)app->userData)->window;
				auto eventCallback = wnd->getEventCallback();

				int width = ANativeWindow_getWidth(app->window);
				int height = ANativeWindow_getHeight(app->window);
				(void)eventCallback->onWindowResized(wnd.get(), width, height);
				break;
			}
			}
		}
	public:
		template<typename android_app_class, typename user_data_type, typename window_event_callback>
		static void androidMain(android_app* app)
		{
			static_assert(std::is_base_of_v<nbl::system::IApplicationFramework::IUserData, user_data_type>);
			system::path CWD = std::filesystem::current_path().generic_string();
			user_data_type engine{};
			nbl::ui::CGraphicalApplicationAndroid::SGraphicalContext ctx{};
			ctx.userData = &engine;
			app->userData = &ctx;
			auto framework = nbl::core::make_smart_refctd_ptr<android_app_class>(app, CWD);
			auto wndManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerAndroid>(app);
			nbl::ui::IWindow::SCreationParams params;
			params.callback = nullptr;
			auto system = core::make_smart_refctd_ptr<nbl::system::CSystemAndroid>(core::make_smart_refctd_ptr<nbl::system::CSystemCallerPOSIX>(), app->activity);
			engine.setSystem(std::move(system));
			if (app->savedState != nullptr) {
					ctx.state = (nbl::system::CApplicationAndroid::SSavedState*)app->savedState;
			}
			android_poll_source* source;
			int ident;
			int events;
			while (framework->keepRunning(app)) {
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
					if (app->window != nullptr)
						framework->workLoopBody(app); 
			}
		}
	};
}

// ... are the window event callback optional ctor params;
#define NBL_ANDROID_MAIN_FUNC(android_app_class, user_data_type, window_event_callback, ...) void android_main(android_app* app){\
		nbl::ui::CGraphicalApplicationAndroid::androidMain<android_app_class, user_data_type, window_event_callback>(app);\
    }

#endif
#endif