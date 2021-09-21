#ifndef	_NBL_SYSTEM_C_APPLICATION_FRAMEWORK_ANDROID_H_INCLUDED_
#define	_NBL_SYSTEM_C_APPLICATION_FRAMEWORK_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/core/declarations.h"
#include "nbl/system/CStdoutLoggerAndroid.h"
#include "nbl/system/IApplicationFramework.h"
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>
namespace nbl::system
{

    class CApplicationAndroid : public IApplicationFramework
    {
    public:
        void onStateSaved(android_app* params)
        {
            return onStateSaved_impl(params);
        }
    protected:
        virtual void onStateSaved_impl(android_app* params) {}

    public:
        struct SSavedState {
            float angle;
            int32_t x;
            int32_t y;
        };
        struct SContext
        {
            SSavedState* state;
            CApplicationAndroid* framework;
            core::smart_refctd_ptr<nbl::ui::IWindow> window;
            void* userData;
        };
    public:
        CApplicationAndroid(android_app* params) : eventPoller(params, this)
        {
            params->onAppCmd = handleCommand;
            params->onInputEvent = handleInput;
            ((SContext*)params->userData)->framework = this;
        }

        static int32_t handleInput(android_app* app, AInputEvent* event) {
            auto* framework = ((SContext*)app->userData)->framework;
            SContext* engine = (SContext*)app->userData;
            if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION) {
                engine->state->x = AMotionEvent_getX(event, 0);
                engine->state->y = AMotionEvent_getY(event, 0);
                return 1;
            }
            return 0;
        }
        static void handleCommand(android_app* app, int32_t cmd) {
            auto* framework = ((SContext*)app->userData)->framework;
            auto* usrData = (SContext*)app->userData;
            auto wnd = ((SContext*)app->userData)->window;
            auto eventCallback = wnd->getEventCallback();
            switch (cmd) {
            case APP_CMD_SAVE_STATE:
                // The system has asked us to save our current state.  Do so.
                usrData->state = (SSavedState*)malloc(sizeof(SSavedState));
                *((SSavedState*)app->savedState) = *usrData->state;
                app->savedStateSize = sizeof(SSavedState);
                framework->onStateSaved(app);
                break;
            case APP_CMD_INIT_WINDOW:
                //debug_break();
                // The window is being shown, get it ready.
               /* if (app->window != nullptr) {
                    engine_init_display(engine);
                    engine_draw_frame(engine);
                }*/
                framework->onAppInitialized(usrData);
                break;
            case APP_CMD_TERM_WINDOW:
                // The window is being hidden or closed, clean it up.
                //engine_term_display(engine);
                framework->onAppTerminated(usrData);
                (void)eventCallback->onWindowClosed(wnd.get());
                break;
            case APP_CMD_WINDOW_RESIZED:
            {
                int width = ANativeWindow_getWidth(app->window);
                int height = ANativeWindow_getHeight(app->window);
                (void)eventCallback->onWindowResized(wnd.get(), width, height);
                break;
            }
            default:
                break;
            }
        }

        class CEventPoller : public  system::IThreadHandler<CEventPoller>
        {
            using base_t = system::IThreadHandler<CEventPoller>;
            friend base_t;
            android_poll_source* source;
            android_app* app;
            ALooper* looper;
            CApplicationAndroid* framework;
            int ident;
            int events;
            bool keepPolling = true;
        public:
            CEventPoller(android_app* _app, CApplicationAndroid* _framework) : app(app), framework(_framework) { }
        protected:
            void init() {
                looper = ALooper_prepare(0); // prepare the looper to poll in the current thread
            }
            void work(typename base_t::lock_t& lock)
            {
                ident = ALooper_pollAll(0, nullptr, &events, (void**)&source);
                if (ident >= 0)
                {
                    if (source != nullptr)
                    {
                        source->process(app, source);
                    }
                    if (app->destroyRequested != 0)
                    {
                        framework->onAppTerminated(app);
                    }
                }
                else keepPolling = false;
            }
            void exit() {}
            bool wakeupPredicate() const { return true; }
        public:
            bool continuePredicate() const { return keepPolling; }
        };
        CEventPoller eventPoller;
        bool keepPolling() const { return eventPoller.continuePredicate(); }
    };

}
// ... are the window event callback optional ctor params;
#define NBL_ANDROID_MAIN(android_app_class, user_data_type, window_event_callback, ...) void android_main(android_app* app){\
    user_data_type engine{};\
    nbl::system::CApplicationAndroid::SContext ctx{};\
    ctx.userData = &engine;\
    app->userData = &ctx;\
    auto framework = nbl::core::make_smart_refctd_ptr<android_app_class>(app);\
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