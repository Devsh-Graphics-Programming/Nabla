#ifndef _NBL_VIDEO_C_VULKAN_SEMAPHORE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_SEMAPHORE_H_INCLUDED_


#include "nbl/video/ISemaphore.h"

#include <volk.h>


namespace nbl::video
{

class ILogicalDevice;

class CVulkanSemaphore final : public ISemaphore
{
    public:
        inline CVulkanSemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& _vkdev, SCreationParams&& creationParams, const VkSemaphore semaphore, const external_handle_t externalHandle)
            : ISemaphore(std::move(_vkdev), std::move(creationParams)), m_semaphore(semaphore), m_externalHandle(externalHandle) {}
        ~CVulkanSemaphore();

        uint64_t getCounterValue() const override;
        void signal(const uint64_t value) override;
    
	    inline const void* getNativeHandle() const override {return &m_semaphore;}
        VkSemaphore getInternalObject() const {return m_semaphore;}
        external_handle_t getExternalHandle() const override { return m_externalHandle; }

        void setObjectDebugName(const char* label) const override;

    private:
        const VkSemaphore m_semaphore;
        const external_handle_t m_externalHandle;
};

}

#endif