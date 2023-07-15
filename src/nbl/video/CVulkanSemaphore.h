#ifndef __NBL_C_VULKAN_SEMAPHORE_H_INCLUDED__
#define __NBL_C_VULKAN_SEMAPHORE_H_INCLUDED__

#include "nbl/video/IGPUSemaphore.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanSemaphore final : public IGPUSemaphore
{
public:
    CVulkanSemaphore(
        core::smart_refctd_ptr<ILogicalDevice>&& _vkdev, 
        SCreationParams&& params,
        VkSemaphore semaphore) 
        : IGPUSemaphore(std::move(_vkdev), std::move(params))
        , m_semaphore(semaphore)
    {}

    ~CVulkanSemaphore();
    
	inline void* getNativeHandle() override {return &m_semaphore;}
    VkSemaphore getInternalObject() const {return m_semaphore;}

    void setObjectDebugName(const char* label) const override;

private:
    VkSemaphore m_semaphore;
};

}

#endif