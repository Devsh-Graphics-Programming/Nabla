#ifndef _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#define _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CApplicationAndroid.h"

namespace nbl::ui
{
	class CGraphicalApplicationAndroid : public system::CApplicationAndroid
	{
	public:
		struct SGraphicalContext : SContext
		{
			core::smart_refctd_ptr<nbl::ui::IWindow> window;
		};
		CGraphicalApplicationAndroid(android_app* app, const system::path& cwd) : system::CApplicationAndroid(app, cwd) {}
	private:
		void handleCommand_impl(android_app* app, int32_t cmd) override
		{
			switch (cmd)
			{
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
	};
}

// ... are the window event callback optional ctor params;
#define NBL_ANDROID_MAIN(android_app_class, user_data_type, window_event_callback, ...) void android_main(android_app* app){\
	system::path CWD = std::filesystem::current_path().generic_string();\
    user_data_type engine{};\
    nbl::ui::CGraphicalApplicationAndroid::SGraphicalContext ctx{};\
    ctx.userData = &engine;\
    app->userData = &ctx;\
    auto framework = nbl::core::make_smart_refctd_ptr<android_app_class>(app, CWD);\
    auto wndManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerAndroid>(app);\
    nbl::ui::IWindow::SCreationParams params;\
    params.callback = nbl::core::make_smart_refctd_ptr<window_event_callback>(__VA_ARGS__);\
    auto wnd = wndManager->createWindow(std::move(params));\
    ctx.window = core::smart_refctd_ptr(wnd);\
    if (app->savedState != nullptr) {\
        ctx.state = (nbl::system::CApplicationAndroid::SSavedState*)app->savedState;\
    }\
    while (framework->keepPolling()) {\
    framework->workLoopBody(app);\
    }\
}
#endif
#endif