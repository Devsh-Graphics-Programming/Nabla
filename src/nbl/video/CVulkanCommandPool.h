#ifndef __NBL_C_VULKAN_COMMAND_POOL_H_INCLUDED__

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"

#include <mutex>

#include <volk.h>

namespace nbl::video
{

class CVulkanCommandPool final : public IGPUCommandPool
{
public:
    CVulkanCommandPool(core::smart_refctd_ptr<ILogicalDevice>&& dev, core::bitflag<IGPUCommandPool::E_CREATE_FLAGS> flags, uint32_t queueFamilyIndex, VkCommandPool vk_commandPool)
        : IGPUCommandPool(std::move(dev), flags.value, queueFamilyIndex), m_vkCommandPool(vk_commandPool)
    {}
    
	inline const void* getNativeHandle() const override {return &m_vkCommandPool;}
    VkCommandPool getInternalObject() const {return m_vkCommandPool;}

    ~CVulkanCommandPool();
	
    void setObjectDebugName(const char* label) const override;

private:
    bool reset_impl() override;

    VkCommandPool m_vkCommandPool;
};

}
#define __NBL_C_VULKAN_COMMAND_POOL_H_INCLUDED__
#endif
