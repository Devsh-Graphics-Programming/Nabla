#ifndef _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#define _NBL_UI_C_GRAPHICAL_APPLICATION_ANDROID_H_INCLUDED_
#ifdef _NBL_PLATFORM_ANDROID_
#include "nbl/system/CApplicationAndroid.h"
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"
#include "nbl/system/CSystemCallerPOSIX.h"
#include "nbl/ui/IGraphicalApplicationFramework.h"
#include "nbl/ui/IWindow.h"
#include "nbl/ui/CWindowManagerAndroid.h"
#include <jni.h>
#include <fstream>
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
		CGraphicalApplicationAndroid(android_app* app, 
			const system::path& _localInputCWD,
			const system::path& _localOutputCWD,
			const system::path& _sharedInputCWD,
			const system::path& _sharedOutputCWD) : system::CApplicationAndroid(app, _localInputCWD, _localOutputCWD, _sharedInputCWD, _sharedOutputCWD) {}
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
		static system::path getSharedResourcesPath(JNIEnv* env);
		public:
			template<typename android_app_class, typename window_event_callback, typename ... EventCallbackArgs>
			static void androidMain(android_app * app, EventCallbackArgs... args)
			{
				JNIEnv* env;
				app->activity->vm->AttachCurrentThread(&env, nullptr);
				system::path sharedInputCWD = system::path(app->activity->externalDataPath).parent_path().parent_path() / "eu.devsh.mediaunpackingonandroid/files/media";
				system::path APKResourcesPath = "asset"; // an archive alias to recognize this path as an apk resource
				system::path sharedOutputCWD = system::path(app->activity->externalDataPath).parent_path().parent_path();
				system::path privateOutputCWD = system::path(app->activity->internalDataPath);


				//std::ofstream ofs(privateOutputCWD / "out.txt");
				//assert(ofs.is_open());
				//ofs << "Hello";
				//ofs.close();

				/*FILE* f = fopen((sharedInputCWD / "dwarf.jpg").string().c_str(), "r");
				std::ifstream ifs(sharedInputCWD / "dwarf.jpg");
				std::string someData;
				bool ex = std::filesystem::exists(sharedInputCWD / "dwarf.jpg");
				assert(ifs.is_open());
				ifs >> someData;
				ifs.close();
				auto mgr = app->activity->assetManager;
				AAssetDir* assetDir = AAssetManager_openDir(mgr, "");
				const char* filename = (const char*)NULL;
				while ((filename = AAssetDir_getNextFileName(assetDir)) != NULL) {
					AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_STREAMING);
					char buf[BUFSIZ];
					int nb_read = 0;
					FILE* out = fopen(filename, "w");
					while ((nb_read = AAsset_read(asset, buf, BUFSIZ)) > 0)
						fwrite(buf, nb_read, 1, out);
					fclose(out);
					AAsset_close(asset);
				}*/

				nbl::ui::CGraphicalApplicationAndroid::SGraphicalContext ctx{};
				app->userData = &ctx;
				auto framework = nbl::core::make_smart_refctd_ptr<android_app_class>(app, APKResourcesPath, privateOutputCWD, sharedInputCWD, sharedOutputCWD);
				auto eventCallback = nbl::core::make_smart_refctd_ptr<window_event_callback>(std::forward<EventCallbackArgs>(args)...);
				auto wndManager = nbl::core::make_smart_refctd_ptr<nbl::ui::CWindowManagerAndroid>(app);
				ctx.wndManager = core::smart_refctd_ptr(wndManager);
				ctx.callback = core::smart_refctd_ptr(eventCallback);
				nbl::ui::IWindow::SCreationParams params;
				params.callback = nullptr;
				auto caller = core::make_smart_refctd_ptr<nbl::system::CSystemCallerPOSIX>();
				auto system = core::make_smart_refctd_ptr<nbl::system::CSystemAndroid>(std::move(caller), app->activity, APKResourcesPath);
				framework->setSystem(std::move(system));
				if (app->savedState != nullptr) {
					ctx.state = *(nbl::system::CApplicationAndroid::SSavedState*)app->savedState;
				}
				android_poll_source* source;
				int ident;
				int events;
				while (framework->getWindow() == nullptr)
				{
					while ((ident = ALooper_pollAll(0, nullptr, &events, (void**)&source)) >= 0)
					{
						if (source != nullptr) {
							source->process(app, source);
						}
					}
				}
				{
					auto wnd = (CWindowAndroid*)framework->getWindow();
					auto mouseChannel = core::make_smart_refctd_ptr<IMouseEventChannel>(CWindowAndroid::CIRCULAR_BUFFER_CAPACITY);
					auto keyboardChannel = core::make_smart_refctd_ptr<IKeyboardEventChannel>(CWindowAndroid::CIRCULAR_BUFFER_CAPACITY);
					if (wnd->addMouseEventChannel(0, core::smart_refctd_ptr<IMouseEventChannel>(mouseChannel)))
						wnd->getEventCallback()->onMouseConnected(wnd, std::move(mouseChannel));
					if (wnd->addKeyboardEventChannel(2, core::smart_refctd_ptr<IKeyboardEventChannel>(keyboardChannel)))
						wnd->getEventCallback()->onKeyboardConnected(wnd, std::move(keyboardChannel));
				}
				while (framework->keepRunning()) {
					while ((ident = ALooper_pollAll(0, nullptr, &events, (void**)&source)) >= 0)
					{
						if (source != nullptr) {
							source->process(app, source);
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