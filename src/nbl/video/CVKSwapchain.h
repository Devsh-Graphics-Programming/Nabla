#ifndef __NBL_C_VK_SWAPCHAIN_H_INCLUDED__
#define __NBL_C_VK_SWAPCHAIN_H_INCLUDED__

#include "nbl/video/ISwapchain.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVKSwapchain final : public ISwapchain
{
public:
    CVKSwapchain(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, SCreationParams&& params,
        images_array_t&& images, VkSwapchainKHR swapchain)
        : ISwapchain(std::move(logicalDevice), std::move(params), std::move(images)),
        m_vkSwapchainKHR(swapchain)
    {}

    ~CVKSwapchain();

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

// Todo(achal): Remove
// private:
    VkSwapchainKHR m_vkSwapchainKHR;
};

}

#endif