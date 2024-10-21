#ifndef	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_
#define	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_


#include "nbl/core/declarations.h"

#include "nbl/system/declarations.h"
#include "nbl/system/definitions.h"

#if defined(_NBL_PLATFORM_WINDOWS_)
#include "nbl/system/CColoredStdoutLoggerWin32.h"
#elif defined(_NBL_PLATFORM_ANDROID_)
#include "nbl/system/CStdoutLoggerAndroid.h"
#endif
#include "nbl/system/CSystemAndroid.h"
#include "nbl/system/CSystemLinux.h"
#include "nbl/system/CSystemWin32.h"

namespace nbl::system
{

class IApplicationFramework : public core::IReferenceCounted
{
	public:
        // this is safe to call multiple times
        static bool GlobalsInit()
        {
            #ifdef _NBL_PLATFORM_WINDOWS_
                #ifdef NBL_CPACK_PACKAGE_DXC_DLL_DIR
                    #ifdef NBL_CPACK_NO_BUILD_DIRECTORY_MODULES
                        const HRESULT dxcLoad = CSystemWin32::delayLoadDLL("dxcompiler.dll", { NBL_CPACK_PACKAGE_DXC_DLL_DIR });
                    #else
                        const HRESULT dxcLoad = CSystemWin32::delayLoadDLL("dxcompiler.dll", { path(_DXC_DLL_).parent_path(), NBL_CPACK_PACKAGE_DXC_DLL_DIR });
                    #endif
                #else
                    const HRESULT dxcLoad = CSystemWin32::delayLoadDLL("dxcompiler.dll", { path(_DXC_DLL_).parent_path() });
                #endif

                if (FAILED(dxcLoad))
                    return false;
                
                #ifdef _NBL_SHARED_BUILD_
                    // if there was no DLL next to the executable, then try from the Nabla build directory
                    // else if nothing in the build dir, then try looking for Nabla in the CURRENT BUILD'S INSTALL DIR
                    // and in CPack package install directory
                
                    #ifdef NBL_CPACK_PACKAGE_NABLA_DLL_DIR
                        #ifdef NBL_CPACK_NO_BUILD_DIRECTORY_MODULES
                            const HRESULT nablaLoad = CSystemWin32::delayLoadDLL(_NABLA_DLL_NAME_, { _NABLA_INSTALL_DIR_, NBL_CPACK_PACKAGE_NABLA_DLL_DIR });
                        #else
                            const HRESULT nablaLoad = CSystemWin32::delayLoadDLL(_NABLA_DLL_NAME_, { _NABLA_OUTPUT_DIR_,_NABLA_INSTALL_DIR_, NBL_CPACK_PACKAGE_NABLA_DLL_DIR });
                        #endif
                    #else
                        const HRESULT nablaLoad = CSystemWin32::delayLoadDLL(_NABLA_DLL_NAME_, { _NABLA_OUTPUT_DIR_,_NABLA_INSTALL_DIR_ });
                    #endif

                    if (FAILED(nablaLoad))
                        return false;
                #endif // _NBL_SHARED_BUILD_
            #else
            // nothing else needs to be done cause we have RPath
            #endif

            return true;
        }

        // we take the derived class as Curiously Recurring Template Parameter
        template<class CRTP> requires std::is_base_of_v<IApplicationFramework,CRTP>
        static int main(int argc, char** argv)
        {
            path CWD = system::path(argv[0]).parent_path().generic_string() + "/";
            auto app = core::make_smart_refctd_ptr<CRTP>(CWD/"../",CWD,CWD/"../../media/",CWD/"../../tmp/");
            for (auto i=0; i<argc; i++)
                app->argv.emplace_back(argv[i]);

            if (!app->onAppInitialized(nullptr))
                return -1;
            while (app->keepRunning())
                app->workLoopBody();
           return app->onAppTerminated() ? 0:(-2);
        }

        static nbl::core::smart_refctd_ptr<ISystem> createSystem()
        {
            if (!GlobalsInit())
                return nullptr;

            #ifdef _NBL_PLATFORM_WINDOWS_
                return nbl::core::make_smart_refctd_ptr<CSystemWin32>();
            #elif defined(_NBL_PLATFORM_ANDROID_)
                return nullptr;
            #endif
            return nullptr;
        }

        // needs to be public because of how constructor forwarding works
        IApplicationFramework(const path& _localInputCWD, const path& _localOutputCWD, const path& _sharedInputCWD, const path& _sharedOutputCWD) :
            localInputCWD(_localInputCWD), localOutputCWD(_localOutputCWD), sharedInputCWD(_sharedInputCWD), sharedOutputCWD(_sharedOutputCWD) 
        {
            const bool status = GlobalsInit();
            assert(status);
        }

        // DEPRECATED
        virtual void setSystem(core::smart_refctd_ptr<ISystem>&& system) {}

        // Some platforms are weird, and you can't really do anything with the system unless you have some magical object that gets passed to the platform's entry point.
        // Therefore the specific `CPLATFORMSystem` is uncreatable out of thin air so you need to take an outside provided one
        virtual bool onAppInitialized(core::smart_refctd_ptr<ISystem>&& system=nullptr) {setSystem(std::move(system)); onAppInitialized_impl(); return true;}
        virtual bool onAppTerminated() {return true;}

        virtual void workLoopBody() = 0;
        virtual bool keepRunning() = 0;

    protected:
        // need this one for skipping the whole constructor chain
        IApplicationFramework() = default;
        virtual ~IApplicationFramework() {}

        // DEPRECATED
        virtual void onAppInitialized_impl() {assert(false);}
        virtual void onAppTerminated_impl() {assert(false);}

        // for platforms with cmdline args
        core::vector<std::string> argv;

        /*
         ****************** Current Working Directories ********************
         Each Nabla app has 4 pre-defined working directories. You can change them to your liking.
         *******************************************************************
        */
        
        
        /*
            This is a CWD which is used for reading app-local assets.
            Do NOT try writing to this path if you wan't your app to work on Android because on Android this CWD is located inside a readonly APK archive.

            To add files to your assets directory, create an "assets" directory in your app's source directory
        */
        path localInputCWD;

        /*
            This is a CWD used to output app-local data e.g. screenshots
        */
        path localOutputCWD;

        /*
            The CWD for input data that can be shared among apps, like the "examples_tests/media" directory for Nabla examples
        */
        path sharedInputCWD;

        /*
            This CWD is used to output data that can be shared between apps e.g. quantization cache
        */
        path sharedOutputCWD;
};

}

// get the correct entry point to declate
#ifdef _NBL_PLATFORM_ANDROID_
    #include "nbl/system/CApplicationAndroid.h"
#else
    #define NBL_MAIN_FUNC(AppClass) int main(int argc, char** argv) \
    {\
		return AppClass::main<AppClass>(argc,argv);\
    }
#endif

#endif