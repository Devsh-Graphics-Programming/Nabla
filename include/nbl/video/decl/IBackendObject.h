#ifndef _NBL_VIDEO_DECLARATION_I_BACKEND_OBJECT_H_INCLUDED_
#define _NBL_VIDEO_DECLARATION_I_BACKEND_OBJECT_H_INCLUDED_


#include "nbl/core/IReferenceCounted.h"

#include <type_traits>

#include "nbl/video/EApiType.h"


namespace nbl::video
{

class ILogicalDevice;
class IQueue;

class IBackendObject : public virtual core::IReferenceCounted
{
    public:
        constexpr static inline size_t MAX_DEBUG_NAME_LENGTH = 255ull;

        E_API_TYPE getAPIType() const;

        bool isCompatibleDevicewise(const IBackendObject* other) const;

        bool wasCreatedBy(const ILogicalDevice* device) const;

        // returns nullptr if `base` is not compatible with device
        template<typename derived_t_ptr, typename base_t_ptr>
        static inline derived_t_ptr device_compatibility_cast(base_t_ptr base, const ILogicalDevice* device)
        {
            using base_t = std::remove_pointer_t<base_t_ptr>;
            using derived_t = std::remove_pointer_t<derived_t_ptr>;
            static_assert(std::is_same_v<std::remove_pointer_t<base_t>,IQueue> || std::is_base_of_v<IBackendObject,base_t>,"base_t should be derived from IBackendObject");
            static_assert(std::is_base_of_v<base_t,derived_t>,"derived_t should be derived from base_t");
            if (base && !base->wasCreatedBy(device))
                return nullptr;
            return static_cast<derived_t_ptr>(base);
        }
        
        // returns nullptr if `base` is not compatible with `other`
        template<typename derived_t_ptr, typename base_t_ptr, typename other_t_ptr>
        static inline derived_t_ptr compatibility_cast(base_t_ptr base, const other_t_ptr& compatibleWith)
        {
            using other_t = std::remove_pointer_t<other_t_ptr>;
            static_assert(std::is_base_of_v<IBackendObject, other_t>,"other_t should be derived from IBackendObject");
            return device_compatibility_cast<derived_t_ptr,base_t_ptr>(base,compatibleWith->getOriginDevice());
        }

        const ILogicalDevice* getOriginDevice() const;

        // to get useful debug messages and names in Renderdoc captures
        virtual void setObjectDebugName(const char* label) const 
        {
            char* out = m_debugName;
            const char* end = m_debugName+MAX_DEBUG_NAME_LENGTH;

            if (label)
            for (const char* in=label; out!=end && *in; in++)
                *(out++) = *in;

            while (out!=end)
                *(out++) = 0;
        }

        const char* getObjectDebugName() const { return m_debugName; }

        // TODO: consider setting tags for backend objects: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkSetDebugUtilsObjectTagEXT.html

    protected:
        IBackendObject(core::smart_refctd_ptr<const ILogicalDevice>&& device);
        virtual ~IBackendObject() = default;

    private:
        const core::smart_refctd_ptr<const ILogicalDevice> m_originDevice;

        mutable char m_debugName[MAX_DEBUG_NAME_LENGTH+1u];
};

}

#endif
