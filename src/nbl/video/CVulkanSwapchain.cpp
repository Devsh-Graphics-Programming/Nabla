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

core::smart_refctd_ptr<CVulkanSwapchain> CVulkanSwapchain::create(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice, ISwapchain::SCreationParams&& params)
{
    constexpr uint32_t MAX_SWAPCHAIN_IMAGE_COUNT = 100u;
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
    vk_createInfo.imageSharingMode = static_cast<VkSharingMode>(params.imageSharingMode);
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

    assert(imageCount <= MAX_SWAPCHAIN_IMAGE_COUNT);

    VkImage vk_images[MAX_SWAPCHAIN_IMAGE_COUNT];
    retval = device->getFunctionTable()->vk.vkGetSwapchainImagesKHR(device->getInternalObject(), vk_swapchain, &imageCount, vk_images);
    if ((retval != VK_SUCCESS) && (retval != VK_INCOMPLETE))
        return nullptr;

    ISwapchain::images_array_t images = core::make_refctd_dynamic_array<ISwapchain::images_array_t>(imageCount);

    uint32_t i = 0u;
    for (auto& image : (*images))
    {
        CVulkanForeignImage::SCreationParams creationParams;
        creationParams.flags = static_cast<CVulkanForeignImage::E_CREATE_FLAGS>(0); // Todo(achal)
        creationParams.type = CVulkanImage::ET_2D;
        creationParams.format = params.surfaceFormat.format;
        creationParams.extent = { params.width, params.height, 1u };
        creationParams.mipLevels = 1u;
        creationParams.arrayLayers = params.arrayLayers;
        creationParams.samples = CVulkanImage::ESCF_1_BIT; // Todo(achal)
        creationParams.usage = params.imageUsage;

        image = core::make_smart_refctd_ptr<CVulkanForeignImage>(
            device, std::move(creationParams),
            vk_images[i++]);
    }

    return core::make_smart_refctd_ptr<CVulkanSwapchain>(
        device, std::move(params),
        std::move(images), vk_swapchain);
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