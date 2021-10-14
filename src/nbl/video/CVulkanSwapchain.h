#ifndef __NBL_C_VULKAN_SWAPCHAIN_H_INCLUDED__
#define __NBL_C_VULKAN_SWAPCHAIN_H_INCLUDED__

#include "nbl/video/ISwapchain.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanSwapchain final : public ISwapchain
{
public:
    CVulkanSwapchain(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, SCreationParams&& params,
        images_array_t&& images, VkSwapchainKHR swapchain)
        : ISwapchain(std::move(logicalDevice), std::move(params), std::move(images)),
        m_vkSwapchainKHR(swapchain)
    {}

    ~CVulkanSwapchain();

    inline VkSwapchainKHR getInternalObject() const { return m_vkSwapchainKHR; }

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;
	
    void setObjectDebugName(const char* label) const override;

// Todo(achal): Remove
// private:
    VkSwapchainKHR m_vkSwapchainKHR;
};

}

#endif