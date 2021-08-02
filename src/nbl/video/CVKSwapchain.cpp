#include "nbl/video/CVKSwapchain.h"

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/surface/ISurfaceVK.h"
#include "nbl/video/CVulkanImage.h"
#include "nbl/video/CVulkanConnection.h"

namespace nbl::video
{

CVKSwapchain::CVKSwapchain(SCreationParams&& params, CVKLogicalDevice* dev)
    : ISwapchain(dev, std::move(params)), m_device(dev)
{
    VkSwapchainCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    createInfo.surface = static_cast<ISurfaceVK*>(m_params.surface.get())->getInternalObject();
    createInfo.minImageCount = m_params.minImageCount;
    createInfo.imageFormat = ISurfaceVK::getVkFormat(m_params.surfaceFormat.format);
    createInfo.imageColorSpace = ISurfaceVK::getVkColorSpaceKHR(m_params.surfaceFormat.colorSpace);
    createInfo.imageExtent = { m_params.width, m_params.height };
    createInfo.imageArrayLayers = m_params.arrayLayers;

    // Todo(achal): Probably need make an enum for this
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    createInfo.imageSharingMode = static_cast<VkSharingMode>(m_params.imageSharingMode);
    createInfo.queueFamilyIndexCount = static_cast<uint32_t>(m_params.queueFamilyIndices->size());
    createInfo.pQueueFamilyIndices = m_params.queueFamilyIndices->data();
    createInfo.preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(m_params.preTransform);

    // Todo(achal): Probably need an enum here
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = static_cast<VkPresentModeKHR>(m_params.presentMode);
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    // auto* vk = m_device->getFunctionTable();
    // VkDevice vkdev = m_device->getInternalObject();

    vkCreateSwapchainKHR(m_device->getInternalObject(), &createInfo, nullptr, &m_swapchain);

    uint32_t imgCount = 0u;
    vkGetSwapchainImagesKHR(m_device->getInternalObject(), m_swapchain, &imgCount, nullptr);
    m_images = core::make_refctd_dynamic_array<images_array_t>(imgCount);

    VkImage vk_Images[100];
    assert(100 >= imgCount);
    vkGetSwapchainImagesKHR(m_device->getInternalObject(), m_swapchain, &imgCount, vk_Images);

    uint32_t i = 0u;
    for (auto& img : (*m_images))
    {
        CVulkanImage::SCreationParams params;
        params.arrayLayers = m_params.arrayLayers;
        params.extent = { m_params.width, m_params.height, 1u };
        params.flags = static_cast<CVulkanImage::E_CREATE_FLAGS>(0);
        params.format = m_params.surfaceFormat.format;
        params.mipLevels = 1u;
        params.samples = CVulkanImage::ESCF_1_BIT;
        params.type = CVulkanImage::ET_2D;
        
        // TODO might want to change this to dev->createImage()
        img = core::make_smart_refctd_ptr<CVulkanImage>(m_device, std::move(params), vk_Images[i++]);
        // img = core::make_smart_refctd_ptr<CVulkanImage>(nullptr, std::move(params), vk_Images[i++]);
    }
}

CVKSwapchain::~CVKSwapchain()
{
    // auto* vk = m_device->getFunctionTable();
    // vk->vk.vkDestroySwapchainKHR(m_device->getInternalObject(), m_swapchain, nullptr);
    vkDestroySwapchainKHR(m_device->getInternalObject(), m_swapchain, nullptr);
}

auto CVKSwapchain::acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) -> E_ACQUIRE_IMAGE_RESULT
{
    // VkDevice dev = m_device->getInternalObject();
    // auto* vk = m_device->getFunctionTable();

    // TODO get semaphore and fence vk handles
    // VkResult result = vk->vk.vkAcquireNextImageKHR(dev, m_swapchain, timeout, 0, 0, out_imgIx);
    VkResult result = vkAcquireNextImageKHR(m_device->getInternalObject(), m_swapchain, timeout, 0, 0, out_imgIx);
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