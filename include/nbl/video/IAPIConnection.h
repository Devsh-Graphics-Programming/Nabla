#ifndef __NBL_I_API_CONNECTION_H_INCLUDED__
#define __NBL_I_API_CONNECTION_H_INCLUDED__

#include "nbl/core/declarations.h"

#include "nbl/video/EApiType.h"
#include "nbl/video/debug/IDebugCallback.h"

namespace nbl::video
{

class IPhysicalDevice;

class IAPIConnection : public core::IReferenceCounted
{
    public:
        virtual E_API_TYPE getAPIType() const = 0;

        virtual core::SRange<IPhysicalDevice *const> getPhysicalDevices() const = 0;

        virtual IDebugCallback* getDebugCallback() const = 0;
};

}


#endif