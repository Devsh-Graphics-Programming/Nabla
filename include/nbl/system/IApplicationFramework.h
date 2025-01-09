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
            // TODO: update CMake and rename "DLL" in all of those defines here to "MODULE" or "RUNTIME"

            auto getEnvInstallDirectory = []()
            {
                const char* sdk = std::getenv("NBL_INSTALL_DIRECTORY");

                if (sdk)
                {
                    const auto directory = system::path(sdk);

                    if (std::filesystem::exists(directory))
                        return directory;
                }

                return system::path("");
            };

            constexpr struct
            {
                std::string_view nabla, dxc;
            } module = 
            {
                #ifdef _NBL_SHARED_BUILD_
                _NABLA_DLL_NAME_
                #else
                ""
                #endif
                , 
                "dxcompiler" 
            };

            const auto sdk = getEnvInstallDirectory();

            struct
            {
                system::path nabla, dxc;
            } install, env, build, rel;

            #if defined(NBL_CPACK_PACKAGE_NABLA_DLL_DIR_ABS_KEY) && defined(NBL_CPACK_PACKAGE_DXC_DLL_DIR_ABS_KEY)
            install.nabla = std::filesystem::absolute(system::path(_NABLA_INSTALL_DIR_) / NBL_CPACK_PACKAGE_NABLA_DLL_DIR_ABS_KEY);
            install.dxc = std::filesystem::absolute(system::path(_NABLA_INSTALL_DIR_) / NBL_CPACK_PACKAGE_DXC_DLL_DIR_ABS_KEY);

            env.nabla = sdk / NBL_CPACK_PACKAGE_NABLA_DLL_DIR_ABS_KEY;
            env.dxc = sdk / NBL_CPACK_PACKAGE_DXC_DLL_DIR_ABS_KEY;
            #endif

            #ifdef _NBL_SHARED_BUILD_
            build.nabla = _NABLA_OUTPUT_DIR_;
            #endif
            build.dxc = path(_DXC_DLL_).parent_path();

            #ifdef NBL_CPACK_PACKAGE_NABLA_DLL_DIR
            rel.nabla = NBL_CPACK_PACKAGE_NABLA_DLL_DIR;
            #endif

            #ifdef NBL_CPACK_PACKAGE_DXC_DLL_DIR
            rel.dxc = NBL_CPACK_PACKAGE_DXC_DLL_DIR;
            #endif

            auto load = [](std::string_view moduleName, const std::vector<system::path>& searchPaths)
            {
                #ifdef _NBL_PLATFORM_WINDOWS_
                const bool isAlreadyLoaded = GetModuleHandleA(moduleName.data());

                if (not isAlreadyLoaded)
                {
                    const HRESULT hook = system::CSystemWin32::delayLoadDLL(moduleName.data(), searchPaths);

                    //! don't be scared if you see "No symbols loaded" - you will not hit "false" in this case, the DLL will get loaded if found,
                    //! proc addresses will be resolved correctly but status may scream "FAILED" due to lack of a PDB to load
                    
                    if (FAILED(hook))
                        return false;
                }
                #else           
                // nothing else needs to be done cause we have RPath 
                // TODO: to be checked when time comes
                #endif

                return true;
            };

            if (not load(module.dxc, { install.dxc, env.dxc, build.dxc, rel.dxc }))
                return false;

            #ifdef _NBL_SHARED_BUILD_
            if (not load(module.nabla, { install.nabla, env.nabla, build.nabla, rel.nabla }))
                return false;
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

            if (not app->isAPILoaded())
                app->onAPILoadFailure();

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
            localInputCWD(_localInputCWD), localOutputCWD(_localOutputCWD), sharedInputCWD(_sharedInputCWD), sharedOutputCWD(_sharedOutputCWD), m_apiLoaded(GlobalsInit()) {}

        virtual bool onAPILoadFailure() { return m_apiLoaded = false; }

        // DEPRECATED
        virtual void setSystem(core::smart_refctd_ptr<ISystem>&& system) {}

        // Some platforms are weird, and you can't really do anything with the system unless you have some magical object that gets passed to the platform's entry point.
        // Therefore the specific `CPLATFORMSystem` is uncreatable out of thin air so you need to take an outside provided one
        virtual bool onAppInitialized(core::smart_refctd_ptr<ISystem>&& system=nullptr) {setSystem(std::move(system)); onAppInitialized_impl(); return true;}
        virtual bool onAppTerminated() {return true;}

        virtual void workLoopBody() = 0;
        virtual bool keepRunning() = 0;

        //! returns status of global initialization - on false you are supposed to terminate the application with non-zero code (otherwise you enter undefined behavior zone)
        inline bool isAPILoaded() { return m_apiLoaded; }

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

        bool m_apiLoaded;
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