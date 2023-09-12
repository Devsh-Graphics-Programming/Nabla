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

core::smart_refctd_ptr<CVulkanSwapchain> CVulkanSwapchain::create(const core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{
    // TODO: assert enabled swapchain/surface feature in logical device, otherwise vkCreateSwapchainKHR is not a valid function pointer and is going to crash 
    if (params.surface->getAPIType() != EAT_VULKAN)
        return nullptr;

    auto device = core::smart_refctd_ptr_static_cast<CVulkanLogicalDevice>(logicalDevice);
    VkSurfaceKHR vk_surface = static_cast<const ISurfaceVulkan*>(params.surface.get())->getInternalObject();
    VkPresentModeKHR vkPresentMode;
    if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_IMMEDIATE) == ISurface::E_PRESENT_MODE::EPM_IMMEDIATE)
        vkPresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    else if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_MAILBOX) == ISurface::E_PRESENT_MODE::EPM_MAILBOX)
        vkPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    else if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_FIFO) == ISurface::E_PRESENT_MODE::EPM_FIFO)
        vkPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    else if ((params.presentMode & ISurface::E_PRESENT_MODE::EPM_FIFO_RELAXED) == ISurface::E_PRESENT_MODE::EPM_FIFO_RELAXED)
        vkPresentMode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;

    VkSwapchainCreateInfoKHR vk_createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    vk_createInfo.surface = vk_surface;
    vk_createInfo.minImageCount = params.minImageCount;
    vk_createInfo.imageFormat = getVkFormatFromFormat(params.surfaceFormat.format);
    vk_createInfo.imageColorSpace = getVkColorSpaceKHRFromColorSpace(params.surfaceFormat.colorSpace);
    vk_createInfo.imageExtent = { params.width, params.height };
    vk_createInfo.imageArrayLayers = params.arrayLayers;
    vk_createInfo.imageUsage = static_cast<VkImageUsageFlags>(params.imageUsage.value);
    vk_createInfo.imageSharingMode = params.isConcurrentSharing() ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
    vk_createInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    vk_createInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    vk_createInfo.preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(params.preTransform);
    vk_createInfo.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(params.compositeAlpha);
    vk_createInfo.presentMode = vkPresentMode;
    vk_createInfo.clipped = VK_FALSE;
    vk_createInfo.oldSwapchain = VK_NULL_HANDLE;
    if (params.oldSwapchain && (params.oldSwapchain->getAPIType() == EAT_VULKAN))
        vk_createInfo.oldSwapchain = IBackendObject::device_compatibility_cast<CVulkanSwapchain*>(params.oldSwapchain.get(), device.get())->getInternalObject();

    VkSwapchainKHR vk_swapchain;
    if (device->getFunctionTable()->vk.vkCreateSwapchainKHR(device->getInternalObject(), &vk_createInfo, nullptr, &vk_swapchain) != VK_SUCCESS)
        return nullptr;

    uint32_t imageCount;
    VkResult retval = device->getFunctionTable()->vk.vkGetSwapchainImagesKHR(device->getInternalObject(), vk_swapchain, &imageCount, nullptr);
    if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        return nullptr;

    assert(imageCount <= ISwapchain::MaxImages);

    IDeviceMemoryBacked::SDeviceMemoryRequirements memReqs;
    memReqs.size = 0ull;
    memReqs.memoryTypeBits = 0x0u;
    memReqs.alignmentLog2 = 63u;
    memReqs.prefersDedicatedAllocation = true;
    memReqs.requiresDedicatedAllocation = true;
    IGPUImage::SCreationParams imgParams;
    imgParams.flags = static_cast<CVulkanImage::E_CREATE_FLAGS>(0);
    imgParams.type = CVulkanImage::ET_2D;
    imgParams.format = params.surfaceFormat.format;
    imgParams.extent = { params.width, params.height, 1u };
    imgParams.mipLevels = 1u;
    imgParams.arrayLayers = params.arrayLayers;
    imgParams.samples = CVulkanImage::ESCF_1_BIT;
    imgParams.usage = params.imageUsage;
    imgParams.skipHandleDestroy = true;

    return core::make_smart_refctd_ptr<CVulkanSwapchain>(
        device, std::move(params),
        std::move(imgParams), std::move(memReqs), imageCount,
        vk_swapchain);
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

ISwapchain::E_PRESENT_RESULT CVulkanSwapchain::present(IGPUQueue* queue, const SPresentInfo& info)
{
    auto logicalDevice = getOriginDevice();
    auto* vk = static_cast<const CVulkanLogicalDevice*>(logicalDevice)->getFunctionTable();

    auto vk_queue = static_cast<const CVulkanQueue*>(queue)->getInternalObject();
    
    constexpr uint32_t MaxWaitSemaphores = 100u;

    if (info.waitSemaphoreCount > MaxWaitSemaphores)
        return ISwapchain::EPR_ERROR;
  
    VkSemaphore vk_waitSemaphores[MaxWaitSemaphores];
    for (uint32_t i = 0u; i < info.waitSemaphoreCount; ++i)
    {
        if (info.waitSemaphores[i]->getAPIType() != EAT_VULKAN)
            return ISwapchain::EPR_ERROR;

        vk_waitSemaphores[i] = IBackendObject::device_compatibility_cast<const CVulkanSemaphore*>(info.waitSemaphores[i], logicalDevice)->getInternalObject();
    }

    auto vk_swapchains = m_vkSwapchainKHR;

    VkPresentInfoKHR presentInfo = { VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = info.waitSemaphoreCount;
    presentInfo.pWaitSemaphores = vk_waitSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &vk_swapchains;
    presentInfo.pImageIndices = &info.imgIndex;

    VkResult retval = vk->vk.vkQueuePresentKHR(vk_queue, &presentInfo);
    switch (retval)
    {
    case VK_SUCCESS:
        return ISwapchain::EPR_SUCCESS;
    case VK_SUBOPTIMAL_KHR:
        return ISwapchain::EPR_SUBOPTIMAL;
    default:
        return ISwapchain::EPR_ERROR;
    }
}

ISwapchain::E_PRESENT_RESULT CVulkanSwapchain::present(CThreadSafeGPUQueueAdapter* queue, const SPresentInfo& info)
{
    std::lock_guard g(queue->m);
    return present(queue->getUnderlyingQueue(), info);
}

core::smart_refctd_ptr<IGPUImage> CVulkanSwapchain::createImage(const uint32_t imageIndex)
{
    if (!setImageExists(imageIndex))
        return nullptr;

    // TODO figure out image create path
    // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/4343
    // for (uint32_t i = 0; i < imageCount; i++)
    // {
    //     VkImageSwapchainCreateInfoKHR vk_imgSwapchainCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_SWAPCHAIN_CREATE_INFO_KHR };
    //     vk_imgSwapchainCreateInfo.pNext = nullptr;
    //     vk_imgSwapchainCreateInfo.swapchain = vk_swapchain;
    // 
    //     // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#swapchain-wsi-image-create-info
    //     VkImageCreateInfo vk_imgCreateInfo = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    //     vk_imgCreateInfo.pNext = &vk_imgSwapchainCreateInfo; // there are a lot of extensions
    //     vk_imgCreateInfo.flags = static_cast<VkImageCreateFlags>(0);
    //     vk_imgCreateInfo.imageType = static_cast<VkImageType>(VK_IMAGE_TYPE_2D);
    //     vk_imgCreateInfo.format = getVkFormatFromFormat(params.surfaceFormat.format);
    //     vk_imgCreateInfo.extent = { params.width, params.height, 1 };
    //     vk_imgCreateInfo.mipLevels = 1;
    //     vk_imgCreateInfo.arrayLayers = params.arrayLayers;
    //     vk_imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    //     vk_imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    //     vk_imgCreateInfo.usage = static_cast<VkImageUsageFlags>(params.imageUsage.value);
    //     vk_imgCreateInfo.sharingMode = static_cast<VkSharingMode>(params.imageSharingMode);
    //     vk_imgCreateInfo.queueFamilyIndexCount = params.queueFamilyIndexCount;
    //     vk_imgCreateInfo.pQueueFamilyIndices = params.queueFamilyIndices;
    //     vk_imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // 
    //     device->getFunctionTable()->vk.vkCreateImage(device->getInternalObject(), &vk_imgCreateInfo, nullptr, &vk_images[i]);
    // 
    //     VkBindImageMemorySwapchainInfoKHR vk_bindImgMemorySwapchainInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_SWAPCHAIN_INFO_KHR };
    //     vk_bindImgMemorySwapchainInfo.pNext = nullptr;
    //     vk_bindImgMemorySwapchainInfo.imageIndex = i;
    //     vk_bindImgMemorySwapchainInfo.swapchain = vk_swapchain;
    // 
    //     VkBindImageMemoryInfo vk_bindImgMemoryInfo = { VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO };
    //     vk_bindImgMemoryInfo.pNext = &vk_bindImgMemorySwapchainInfo;
    //     vk_bindImgMemoryInfo.image = vk_images[i];
    //     vk_bindImgMemoryInfo.memory = nullptr;
    // 
    //     device->getFunctionTable()->vk.vkBindImageMemory2(device->getInternalObject(), 1, &vk_bindImgMemoryInfo);
    //     assert(vk_images[i]);
    // }

    auto device = core::smart_refctd_ptr<const CVulkanLogicalDevice>(static_cast<const CVulkanLogicalDevice*>(getOriginDevice()));
    // TODO avoid getting all the images each time
    VkImage vk_images[ISwapchain::MaxImages];
    uint32_t imageCount = m_imageCount;
    auto retval = device->getFunctionTable()->vk.vkGetSwapchainImagesKHR(device->getInternalObject(), m_vkSwapchainKHR, &imageCount, vk_images);
    if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        return nullptr;

    auto creationParams = std::move(m_imgCreationParams);
    std::unique_ptr<CCleanupSwapchainReference> swapchainRef(new CCleanupSwapchainReference{});
    swapchainRef->m_swapchain = core::smart_refctd_ptr<ISwapchain>(this);
    swapchainRef->m_imageIndex = imageIndex;

    creationParams.preDestroyCleanup = std::unique_ptr<ICleanup>(swapchainRef.release());
    creationParams.skipHandleDestroy = true;

    auto image = core::make_smart_refctd_ptr<CVulkanImage>(
        std::move(device),
        m_imgMemRequirements, std::move(creationParams), vk_images[imageIndex]);

    return image;
}

}
