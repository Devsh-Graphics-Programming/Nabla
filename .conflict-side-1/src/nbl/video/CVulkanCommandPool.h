#ifndef _NBL_VIDEO_C_VULKAN_COMMAND_POOL_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_COMMAND_POOL_H_INCLUDED_

#include "nbl/video/IGPUCommandPool.h"
#include "nbl/core/containers/CMemoryPool.h"

#include <mutex>

#include <volk.h>

namespace nbl::video
{

class CVulkanCommandPool final : public IGPUCommandPool
{
    public:
        CVulkanCommandPool(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const core::bitflag<IGPUCommandPool::CREATE_FLAGS> flags, const uint32_t queueFamilyIndex, const VkCommandPool vk_commandPool)
            : IGPUCommandPool(std::move(dev), flags, queueFamilyIndex), m_vkCommandPool(vk_commandPool) {}

        void trim() override;

	    inline const void* getNativeHandle() const override {return &m_vkCommandPool;}
        VkCommandPool getInternalObject() const {return m_vkCommandPool;}
	
        void setObjectDebugName(const char* label) const override;

    private:
        ~CVulkanCommandPool();

        bool createCommandBuffers_impl(const BUFFER_LEVEL level, const std::span<core::smart_refctd_ptr<IGPUCommandBuffer>> outCmdBufs, core::smart_refctd_ptr<system::ILogger>&& logger);

        bool reset_impl() override;

        const VkCommandPool m_vkCommandPool;
};

}
#endif
