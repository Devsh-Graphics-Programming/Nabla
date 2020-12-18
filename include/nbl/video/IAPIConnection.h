#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/surface/ISurfaceWin32.h"

namespace nbl {
namespace video
{

class IAPIConnection : public core::IReferenceCounted
{
public:
    enum E_TYPE
    {
        ET_OPENGL,
        ET_VULKAN
    };

    // TODO implement in some source file in src/nbl/...
    static core::smart_refctd_ptr<IAPIConnection> create(E_TYPE apiType);

#ifdef _NBL_PLATFORM_WINDOWS_
    virtual core::smart_refctd_ptr<ISurfaceWin32> createSurfaceWin32(ISurfaceWin32::SCreationParams&& params) const = 0;
#endif
    //etc... for other platforms...

    virtual E_TYPE getAPIType() const = 0;

protected:
    IAPIConnection() = default;
};

}
}


#endif