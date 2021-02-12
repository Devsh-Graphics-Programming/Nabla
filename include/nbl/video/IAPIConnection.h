#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IPhysicalDevice.h"
#include "nbl/video/EApiType.h"

namespace nbl {
namespace video
{

class IAPIConnection : public core::IReferenceCounted
{
public:
    // TODO implement in some source file in src/nbl/...
    static core::smart_refctd_ptr<IAPIConnection> create(E_API_TYPE apiType, uint32_t appVer, const char* appName);

    virtual E_API_TYPE getAPIType() const = 0;

    virtual core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const = 0;

protected:
    IAPIConnection() = default;
    virtual ~IAPIConnection() = default;
};

}
}


#endif