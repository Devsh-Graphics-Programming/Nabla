#ifndef	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_
#define	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_

#include "nbl/core/declarations.h"

#include "nbl/system/declarations.h"
#include "nbl/system/definitions.h"

#ifdef _NBL_PLATFORM_WINDOWS_
#include <delayimp.h>
#endif // _NBL_PLATFORM_WINDOWS_

namespace nbl::system
{

class NBL_API IApplicationFramework : public core::IReferenceCounted
{
	public:
        virtual void setSystem(core::smart_refctd_ptr<nbl::system::ISystem>&& system) = 0;
        IApplicationFramework(
            const system::path& _localInputCWD, 
            const system::path& _localOutputCWD, 
            const system::path& _sharedInputCWD, 
            const system::path& _sharedOutputCWD) : 
            localInputCWD(_localInputCWD), localOutputCWD(_localOutputCWD), sharedInputCWD(_sharedInputCWD), sharedOutputCWD(_sharedOutputCWD)
		{
#if defined(_NBL_PLATFORM_WINDOWS_) && defined(_NBL_SHARED_BUILD_)
            HMODULE res = LoadLibraryExA(_NABLA_DLL_NAME_, NULL, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
            if (!res)
            {
                const auto nablaBuiltDLL = (system::path(_NABLA_OUTPUT_DIR_).make_preferred() / _NABLA_DLL_NAME_).string();
                res = LoadLibraryExA(nablaBuiltDLL.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            }
            if (!res)
            {
                const auto nablaInstalledDLL = (system::path(_NABLA_INSTALL_DIR_).make_preferred() / _NABLA_DLL_NAME_).string();
                res = LoadLibraryExA(nablaInstalledDLL.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
            }
            if (!res)
                res = LoadLibraryExA(_NABLA_DLL_NAME_, NULL, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            HRESULT hr = __HrLoadAllImportsForDll(_NABLA_DLL_NAME_);
            assert(res && SUCCEEDED(hr));
#endif // _NBL_PLATFORM_WINDOWS_ && _NBL_SHARED_BUILD_
		}

        void onAppInitialized()
        {
            return onAppInitialized_impl();
        }
        void onAppTerminated()
        {
            return onAppTerminated_impl();
        }

        virtual void onResize(uint32_t w, uint32_t h) {}

        virtual void workLoopBody() = 0;
        virtual bool keepRunning() = 0;
        std::vector<std::string> argv;

    protected:
        ~IApplicationFramework() {}

        // TODO: why aren't these pure virtual, and why do we even need a `_impl()`suffix?
        virtual void onAppInitialized_impl() {}
        virtual void onAppTerminated_impl() {}

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
        system::path localInputCWD;

        /*
            This is a CWD used to output app-local data e.g. screenshots
        */
        system::path localOutputCWD;

        /*
            The CWD for input data that can be shared among apps, like the "examples_tests/media" directory for Nabla examples
        */
        system::path sharedInputCWD;

        /*
            This CWD is used to output data that can be shared between apps e.g. quantization cache
        */
        system::path sharedOutputCWD;
};

}

#endif