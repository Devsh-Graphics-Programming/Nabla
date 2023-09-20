#ifndef _NBL_C_VULKAN_SWAPCHAIN_H_INCLUDED_
#define _NBL_C_VULKAN_SWAPCHAIN_H_INCLUDED_

#include "nbl/video/ISwapchain.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;
class CThreadSafeGPUQueueAdapter;

class CVulkanSwapchain final : public ISwapchain
{
public:
    CVulkanSwapchain(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, SCreationParams&& params, 
        IGPUImage::SCreationParams&& imgCreationParams, IDeviceMemoryBacked::SDeviceMemoryRequirements&& imgMemRequirements, uint32_t imageCount,
        VkSwapchainKHR swapchain)
        : ISwapchain(std::move(logicalDevice), std::move(params), std::move(imgCreationParams), imageCount),
        m_vkSwapchainKHR(swapchain), m_imgMemRequirements(std::move(imgMemRequirements))
    {}

    NBL_API2 static core::smart_refctd_ptr<CVulkanSwapchain> create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params);

    ~CVulkanSwapchain();

    inline const void* getNativeHandle() const {return &m_vkSwapchainKHR;}
    inline VkSwapchainKHR getInternalObject() const {return m_vkSwapchainKHR;}

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

    E_PRESENT_RESULT present(IGPUQueue* queue, const SPresentInfo& info) override;

    E_PRESENT_RESULT present(CThreadSafeGPUQueueAdapter* queue, const SPresentInfo& info);

    core::smart_refctd_ptr<IGPUImage> createImage(const uint32_t imageIndex) override;

    void setObjectDebugName(const char* label) const override;

private:
    VkSwapchainKHR m_vkSwapchainKHR;
    IDeviceMemoryBacked::SDeviceMemoryRequirements m_imgMemRequirements;
};

}

#endif