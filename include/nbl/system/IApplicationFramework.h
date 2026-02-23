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
#include "nbl/system/RuntimeModuleLookup.h"

namespace nbl::system
{

class IApplicationFramework : public core::IReferenceCounted
{
	public:
        // this is safe to call multiple times
        static bool GlobalsInit()
        {
            RuntimeModuleLookup lookup;

            const auto exeDirectory = system::executableDirectory();
            lookup.applyInstallOverrides(exeDirectory);
            /*
                In the current design build interface and install interface cannot share one lookup set.

                Build lookup may point to host-only output folders while install lookup must stay relocatable.
                Mixing them can load stale modules from host build trees and break packaged consumers.
                Another big issue is Nabla build-system layout because runtime binaries are emitted into
                source-side locations instead of a binary-tree runtime prefix that mirrors install layout.
                This makes executable-relative lookup ambiguous and forces a split between build and install lookup modes.
                There are more issues caused by this non-unified layout than the ones handled in this file.

                Desired end state is that build outputs follow the same relative runtime layout as install so lookup can stay install-style
                for both host build and package consumers while still allowing consumer override paths like "./Libraries".
                No interface should ever expose any define that contains an absolute path.
                All binaries must be emitted into the build directory and Nabla 
                should remain fully buildable with a read-only source filesystem.

                I cannot address all of that here because it requires a broader Nabla build-system refactor.
            */
            const bool useInstallLookups = lookup.chooseInstallLookupMode(exeDirectory);
            lookup.finalizeInstallLookups(useInstallLookups);

            using SearchPaths = std::vector<system::path>;
            const auto load = [](std::string_view moduleName, const SearchPaths& searchPaths)
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

            if (not load(lookup.dxc.name, useInstallLookups ? SearchPaths{ lookup.dxc.paths.install } : SearchPaths{ lookup.dxc.paths.build }))
                return false;

            #ifdef _NBL_SHARED_BUILD_
            if (not load(lookup.nabla.name, useInstallLookups ? SearchPaths{ lookup.nabla.paths.install } : SearchPaths{ lookup.nabla.paths.build }))
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
