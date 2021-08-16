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
        IBackendObject(const ILogicalDevice* device) : m_originDevice2(device) {}
        IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>&& device);

        E_API_TYPE getAPIType() const;

        bool isCompatibleDevicewise(const IBackendObject* other) const;

        bool wasCreatedBy(const ILogicalDevice* device) const;

    protected:
        virtual ~IBackendObject();
        const ILogicalDevice* getOriginDevice() const;

    private:
        const core::smart_refctd_ptr<const ILogicalDevice> m_originDevice;
        const ILogicalDevice* m_originDevice2;
};

}

#endif
