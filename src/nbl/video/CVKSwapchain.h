#ifndef __NBL_C_VK_SWAPCHAIN_H_INCLUDED__
#define __NBL_C_VK_SWAPCHAIN_H_INCLUDED__

#include <volk.h>
#include "nbl/video/ISwapchain.h"

namespace nbl::video
{

class CVKLogicalDevice;

class CVKSwapchain final : public ISwapchain
{
public:
    CVKSwapchain(SCreationParams&& params, CVKLogicalDevice* dev);
    ~CVKSwapchain();

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

private:
    CVKLogicalDevice* m_device;
    VkSwapchainKHR m_swapchain;
};

}

#endif