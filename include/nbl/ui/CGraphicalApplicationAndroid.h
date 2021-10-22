#ifndef _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#define _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CApplicationAndroid.h"
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"
#include "nbl/system/CSystemCallerPOSIX.h"
#include "nbl/ui/IGraphicalApplicationFramework.h"
#include "nbl/ui/IWindow.h"

namespace nbl::ui
{
	class CGraphicalApplicationAndroid : public system::CApplicationAndroid, public ui::IGraphicalApplicationFramework
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
		void handleInput_impl(android_app* app, AInputEvent* event) override
		{
			auto* ctx = (SGraphicalContext*)app->userData;
			ctx->wndManager->handleInput_impl(app, event);
		}
		void handleCommand_impl(android_app* app, int32_t cmd) override
		{
			auto* ctx = (SGraphicalContext*)app->userData;
			ctx->wndManager->handleCommand_impl(app, cmd);
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