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
        constexpr static inline size_t MAX_DEBUG_NAME_LENGTH = 255ull;

        IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>&& device);

        E_API_TYPE getAPIType() const;

        bool isCompatibleDevicewise(const IBackendObject* other) const;

        bool wasCreatedBy(const ILogicalDevice* device) const;

        const ILogicalDevice* getOriginDevice() const;

        // to get useful debug messages and names in Renderdoc captures
        virtual void setObjectDebugName(const char* label) const 
        {
#ifdef _NBL_DEBUG
            size_t len = strlen(label);
            assert(len <= MAX_DEBUG_NAME_LENGTH);
#endif
            strcpy(m_debugName, label);
        }

        const char* getObjectDebugName() const { return m_debugName; }

    protected:
        virtual ~IBackendObject();

    private:
        const core::smart_refctd_ptr<const ILogicalDevice> m_originDevice;

        mutable char m_debugName[MAX_DEBUG_NAME_LENGTH+1u];
};

}

#endif
