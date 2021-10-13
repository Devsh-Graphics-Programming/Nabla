#ifndef _NBL_SYSTEM_C_WINDOW_MANAGER_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_WINDOW_MANAGER_ANDROID_H_INCLUDED_
#include "nbl/ui/IWindowManager.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>

#include "nbl/ui/CWindowAndroid.h"

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
			return core::make_smart_refctd_ptr<nbl::ui::CWindowAndroid>(std::move(creationParams), m_app->window);
		}
        void destroyWindow(IWindow* wnd) override final
        { 
        }
	};
}

#endif
#endif