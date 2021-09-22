#include "nbl/video/CVulkanSwapchain.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

CVulkanSwapchain::~CVulkanSwapchain()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroySwapchainKHR(vulkanDevice->getInternalObject(), m_vkSwapchainKHR, nullptr);
}

auto CVulkanSwapchain::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) -> E_ACQUIRE_IMAGE_RESULT
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() != EAT_VULKAN)
        return EAIR_ERROR;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(originDevice);
    VkDevice vk_device = vulkanDevice->getInternalObject();
    auto* vk = vulkanDevice->getFunctionTable();

    VkSemaphore vk_semaphore = VK_NULL_HANDLE;
    if (semaphore && semaphore->getAPIType() == EAT_VULKAN)
        vk_semaphore = static_cast<const CVulkanSemaphore*>(semaphore)->getInternalObject();

    VkFence vk_fence = VK_NULL_HANDLE;
    if (fence && fence->getAPIType() == EAT_VULKAN)
        vk_fence = static_cast<const CVulkanFence*>(fence)->getInternalObject();

    VkResult result = vk->vk.vkAcquireNextImageKHR(vk_device, m_vkSwapchainKHR, timeout, vk_semaphore, vk_fence, out_imgIx);

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