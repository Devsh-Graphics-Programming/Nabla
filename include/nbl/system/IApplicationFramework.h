#ifndef _NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_
#define _NBL_SYSTEM_I_APPLICATION_FRAMEWORK_H_INCLUDED_

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
        const system::path& _sharedOutputCWD)
        : localInputCWD(_localInputCWD), localOutputCWD(_localOutputCWD), sharedInputCWD(_sharedInputCWD), sharedOutputCWD(_sharedOutputCWD)
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

    system::path localInputCWD, localOutputCWD, sharedInputCWD, sharedOutputCWD;
};

}

#endif