#ifndef __NBL_VIDEO_DECLARATION_I_BACKEND_OBJECT_H_INCLUDED__
#define __NBL_VIDEO_DECLARATION_I_BACKEND_OBJECT_H_INCLUDED__


#include "nbl/video/EApiType.h"

#include <type_traits>

namespace nbl::video
{

class ILogicalDevice;

class IBackendObject
{
    public:
        IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>&& device);

        E_API_TYPE getAPIType() const;

        bool isCompatibleDevicewise(const IBackendObject* other) const;

        bool wasCreatedBy(const ILogicalDevice* device) const;

        const ILogicalDevice* getOriginDevice() const;

        // to get useful debug messages and names in Renderdoc captures
        virtual void setDebugName(const char* label) {} //=0;

    protected:
        virtual ~IBackendObject();

    private:
        const core::smart_refctd_ptr<const ILogicalDevice> m_originDevice;
};

}

#endif
