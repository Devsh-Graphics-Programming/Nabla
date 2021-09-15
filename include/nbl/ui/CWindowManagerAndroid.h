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
    struct saved_state {
        float angle;
        int32_t x;
        int32_t y;
    };
    struct user_data
    {
        saved_state state;
    };

	class CWindowManagerAndroid : public IWindowManager
	{
		android_app* m_app;
	public:
		CWindowManagerAndroid(android_app* app) : m_app(app) 
        {
            m_app->onAppCmd = engine_handle_cmd;
            m_app->onInputEvent = engine_handle_input;
        }
		~CWindowManagerAndroid() = default;
		core::smart_refctd_ptr<IWindow> createWindow(IWindow::SCreationParams&& creationParams) override final
		{
			return core::make_smart_refctd_ptr<CWindowAndroid>(m_app->window);
		}
        void destroyWindow(IWindow* wnd) override final
        { 
        }
		static int32_t engine_handle_input(android_app* app, AInputEvent* event) {
			LOGI("TEST HERE");
            user_data* engine = (user_data*)app->userData;
			if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
				engine->state.x = AMotionEvent_getX(event, 0);
				engine->state.y = AMotionEvent_getY(event, 0);
				return 1;
			}
			return 0;
		}
        static void engine_handle_cmd(android_app* app, int32_t cmd) {
            //debug_break();
            auto* usrData = (user_data*)app->userData;
            switch (cmd) {
            case APP_CMD_SAVE_STATE:
                // The system has asked us to save our current state.  Do so.
                //usrData->savedState = malloc(sizeof(saved_state));
                //*((saved_state*)app->savedState) = usrData->state;
                //app->savedStateSize = sizeof(saved_state);
                break;
            case APP_CMD_INIT_WINDOW:
                //debug_break();
                // The window is being shown, get it ready.
                if (app->window != nullptr) {
                    //engine_init_display(engine);
                    //engine_draw_frame(engine);
                }
                break;
            case APP_CMD_TERM_WINDOW:
                // The window is being hidden or closed, clean it up.
                //engine_term_display(engine);
                break;
                /*
                case APP_CMD_GAINED_FOCUS:
                    // When our app gains focus, we start monitoring the accelerometer.
                    if (engine->accelerometerSensor != nullptr) {
                        ASensorEventQueue_enableSensor(engine->sensorEventQueue,
                                                       engine->accelerometerSensor);
                        // We'd like to get 60 events per second (in us).
                        ASensorEventQueue_setEventRate(engine->sensorEventQueue,
                                                       engine->accelerometerSensor,
                                                       (1000L/60)*1000);
                    }
                    break;
                case APP_CMD_LOST_FOCUS:
                    // When our app loses focus, we stop monitoring the accelerometer.
                    // This is to avoid consuming battery while not being used.
                    if (engine->accelerometerSensor != nullptr) {
                        ASensorEventQueue_disableSensor(engine->sensorEventQueue,
                                                        engine->accelerometerSensor);
                    }
                    // Also stop animating.
                    engine->animating = 0;
                    engine_draw_frame(engine);
                    break;
                */
            default:
                break;
            }
        }
	};
}

#endif
#endif