#ifndef _NBL_C_VULKAN_SEMAPHORE_H_INCLUDED_
#define _NBL_C_VULKAN_SEMAPHORE_H_INCLUDED_


#include "nbl/video/ISemaphore.h"

#include <volk.h>


namespace nbl::video
{

class ILogicalDevice;

class CVulkanSemaphore final : public ISemaphore
{
    public:
        inline CVulkanSemaphore(core::smart_refctd_ptr<const ILogicalDevice>&& _vkdev, const bool _isTimeline, const VkSemaphore semaphore)
            : ISemaphore(std::move(_vkdev),_isTimeline), m_semaphore(semaphore) {}
        ~CVulkanSemaphore();
    
	    inline const void* getNativeHandle() const override {return &m_semaphore;}
        VkSemaphore getInternalObject() const {return m_semaphore;}

        void setObjectDebugName(const char* label) const override;

    private:
        uint64_t getCounterValue_impl() const override;
        void signal_impl(const uint64_t value) override;

        const VkSemaphore m_semaphore;
};

}

#endif