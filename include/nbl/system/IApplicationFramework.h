#ifndef	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_
#define	_NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_

#include "nbl/core/declarations.h"
#include "nbl/system/declarations.h"

#include "nbl/system/definitions.h"

namespace nbl::system
{

class IApplicationFramework : public core::IReferenceCounted
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
		}

        void onAppInitialized()
        {
            return onAppInitialized_impl();
        }
        void onAppTerminated()
        {
            return onAppTerminated_impl();
        }

        virtual void workLoopBody() = 0;
        virtual bool keepRunning() = 0;
        std::vector<std::string> argv;

    protected:
        ~IApplicationFramework() {}

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