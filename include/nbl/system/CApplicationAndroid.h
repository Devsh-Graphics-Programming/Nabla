#ifndef	_NBL_SYSTEM_C_APPLICATION_FRAMEWORK_ANDROID_H_INCLUDED_
#define	_NBL_SYSTEM_C_APPLICATION_FRAMEWORK_ANDROID_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/system/IApplicationFramework.h"
#include "nbl/system/CStdoutLoggerAndroid.h"

#ifdef _NBL_PLATFORM_ANDROID_
#include <android_native_app_glue.h>
#include <android/sensor.h>
#include <android/log.h>
#endif

namespace nbl::system
{
#ifdef _NBL_PLATFORM_ANDROID_

class CApplicationAndroid : public IApplicationFramework
{
    public:
        void onStateSaved(android_app* params)
        {
            return onStateSaved_impl(params);
        }
    protected:
        virtual void onStateSaved_impl(android_app* params) {}
        bool paused = false;
      /*  bool initialized = false;*/
        android_app* m_app;
        JNIEnv* m_env;
    public:
        void pause() { paused = true; }
        void resume() { paused = false; }
        //void initialize() { initialized = true; }
        //bool isInitialized() const { return initialized; }
        bool isPaused() const { return paused; }
        android_app* getApp() { return m_app; }
        struct SSavedState {
            float angle;
            int32_t x;
            int32_t y;
        };
        struct SContext
        {
            CApplicationAndroid* framework;
            void* userData;
            SSavedState* state;
        };
    public:
        CApplicationAndroid(android_app* params,
            JNIEnv* env,
            const system::path& _localInputCWD,
            const system::path& _localOutputCWD,
            const system::path& _sharedInputCWD,
            const system::path& _sharedOutputCWD
        ) : IApplicationFramework(_localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD),  eventPoller(params, this), m_app(params), m_env(env)
        {
            params->onAppCmd = handleCommand;
            params->onInputEvent = handleInput;
            ((SContext*)params->userData)->framework = this;
        }

        static int32_t handleInput(android_app* app, AInputEvent* event) {
            auto* framework = ((SContext*)app->userData)->framework;
            framework->handleInput_impl(app, event);
            return 0;
        }
        static void handleCommand(android_app* app, int32_t cmd) {
            auto* framework = ((SContext*)app->userData)->framework;
            framework->handleCommand_impl(app, cmd);
            auto* usrData = (SContext*)app->userData;
            switch (cmd) {
            case APP_CMD_SAVE_STATE:
                // The system has asked us to save our current state.  Do so.
                //usrData->state = (SSavedState*)malloc(sizeof(SSavedState));
                //app->savedState = usrData->state;
                //app->savedStateSize = sizeof(SSavedState);
                //framework->onStateSaved(app);
                break;
            case APP_CMD_INIT_WINDOW:
            {
               /* if (!framework->isInitialized())
                {
                    framework->onAppInitialized();
                    framework->initialize();
                }*/

            }
                break;
            default:
                break;
            }
        }
        virtual void handleCommand_impl(android_app* data, int32_t cmd) {}
        virtual void handleInput_impl(android_app* data, AInputEvent* event) {}

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
            CEventPoller(android_app* _app, CApplicationAndroid* _framework) : base_t(base_t::start_on_construction), app(_app), framework(_framework)
            {
                waitForInitComplete();
            }
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
                        framework->onAppTerminated();
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

#endif
}

#endif 