#ifndef _NBL_SYSTEM_C_WINDOW_MANAGER_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_WINDOW_MANAGER_ANDROID_H_INCLUDED_
#include "nbl/ui/IWindowManager.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>

#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", __VA_ARGS__))

namespace nbl::ui
{
	class CWindowManagerAndroid : public IWindowManager
	{
		android_app* m_app;
	public:
		CWindowManagerAndroid(android_app* app) : m_app(app) 
        {
        }
		~CWindowManagerAndroid() = default;
		core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final
		{
			return core::make_smart_refctd_ptr<CWindowAndroid>(m_app->window);
		}
        void destroyWindow(IWindow* wnd) override final
        { 
        }
	};
}

#endif
#endif