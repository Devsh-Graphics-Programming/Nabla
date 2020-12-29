#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IPhysicalDevice.h"

namespace nbl {
namespace video
{

class IAPIConnection : public core::IReferenceCounted
{
public:
    enum E_TYPE
    {
        ET_OPENGL,
        ET_OPENGL_ES,
        ET_VULKAN
    };

    // TODO implement in some source file in src/nbl/...
    static core::smart_refctd_ptr<IAPIConnection> create(E_TYPE apiType);

    virtual E_TYPE getAPIType() const = 0;

    virtual core::SRange<const core::smart_refctd_ptr<IPhysicalDevice>> getPhysicalDevices() const = 0;

protected:
    IAPIConnection() = default;
    virtual ~IAPIConnection() = default;
};

}
}


#endif