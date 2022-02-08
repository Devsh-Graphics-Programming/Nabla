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
        vk_semaphore = IBackendObject::compatibility_cast<const CVulkanSemaphore*>(semaphore, this)->getInternalObject();

    VkFence vk_fence = VK_NULL_HANDLE;
    if (fence && fence->getAPIType() == EAT_VULKAN)
        vk_fence = IBackendObject::compatibility_cast<const CVulkanFence*>(fence, this)->getInternalObject();

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

void CVulkanSwapchain::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(vkSetDebugUtilsObjectNameEXT == 0) return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_SWAPCHAIN_KHR;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}
}