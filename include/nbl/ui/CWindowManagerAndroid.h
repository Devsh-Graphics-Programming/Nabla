#ifndef _NBL_SYSTEM_C_WINDOW_MANAGER_ANDROID_H_INCLUDED_
#define _NBL_SYSTEM_C_WINDOW_MANAGER_ANDROID_H_INCLUDED_
#include "nbl/ui/IWindowManager.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>

#include "nbl/ui/CWindowAndroid.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/fb.h>

namespace nbl::ui
{
	class CWindowManagerAndroid : public IWindowManager
	{
		android_app* m_app;
		std::atomic_flag windowIsCreated;
		constexpr static uint32_t CIRCULAR_BUFFER_CAPACITY = CWindowAndroid::CIRCULAR_BUFFER_CAPACITY;

		float lastCursorX, lastCursorY;
		bool initialized = false;
	public:
		CWindowManagerAndroid(android_app* app) : m_app(app) 
        {
			windowIsCreated.clear();
        }
		~CWindowManagerAndroid() = default;
		core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final
		{
			bool createdBefore = windowIsCreated.test_and_set();
			if (!createdBefore)
			{
				return core::make_smart_refctd_ptr<nbl::ui::CWindowAndroid>(std::move(creationParams), m_app->window);
			}
			return nullptr;
		}
		SDisplayInfo getPrimaryDisplayInfo() const override final
		{
			struct fb_var_screeninfo fb_var;
			int fd = open("/dev/graphics/fb0", O_RDONLY);
			ioctl(fd, FBIOGET_VSCREENINFO, &fb_var);
			close(fd);

			SDisplayInfo info{};
			info.resX = fb_var.xres;
			info.resY = fb_var.yres;
			info.x = fb_var.xoffset;
			info.y = fb_var.yoffset;
			return info;
		}
        void destroyWindow(IWindow* wnd) override final
        { 
        }
		void handleInput_impl(android_app* data, AInputEvent* event);
		void handleCommand_impl(android_app* app, int32_t cmd);
		static E_KEY_CODE getNablaKeyCodeFromNative(int32_t nativeKeyCode);

	};
}

#endif
#endif