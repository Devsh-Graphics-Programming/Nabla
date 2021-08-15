#include "nbl/video/CVKSwapchain.h"

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

CVKSwapchain::~CVKSwapchain()
{
    const auto originDevice = getOriginDevice();

    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        // auto* vk = m_vkdev->getFunctionTable();
        VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        // vk->vk.vkDestroySwapchainKHR(m_device->getInternalObject(), m_swapchain, nullptr);
        vkDestroySwapchainKHR(vk_device, m_vkSwapchainKHR, nullptr);
    }
}

auto CVKSwapchain::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) -> E_ACQUIRE_IMAGE_RESULT
{
    // VkDevice dev = m_device->getInternalObject();
    // auto* vk = m_device->getFunctionTable();

    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() != EAT_VULKAN)
        return EAIR_ERROR;

    VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();

    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
    if (semaphore && semaphore->getAPIType() == EAT_VULKAN)
        vk_semaphore = reinterpret_cast<CVulkanSemaphore*>(semaphore)->getInternalObject();

    VkFence vk_fence = VK_NULL_HANDLE;
    if (fence && fence->getAPIType() == EAT_VULKAN)
        vk_fence = reinterpret_cast<CVulkanFence*>(fence)->getInternalObject();

    // VkResult result = vk->vk.vkAcquireNextImageKHR(dev, m_swapchain, timeout, 0, 0, out_imgIx);
    VkResult result = vkAcquireNextImageKHR(vk_device, m_vkSwapchainKHR, timeout,
        vk_semaphore, vk_fence, out_imgIx);

    switch (result)
    {
    case VK_SUCCESS:
        return EAIR_SUCCESS;
    case VK_TIMEOUT:
        return EAIR_TIMEOUT;
    case VK_NOT_READY:
        return EAIR_NOT_READY;
    case VK_SUBOPTIMAL_KHR:
        return EAIR_SUBOPTIMAL;
    default:
        return EAIR_ERROR;
    }
}

}