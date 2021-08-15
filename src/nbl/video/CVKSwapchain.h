#ifndef __NBL_C_VK_SWAPCHAIN_H_INCLUDED__
#define __NBL_C_VK_SWAPCHAIN_H_INCLUDED__

#include <volk.h>
#include "nbl/video/ISwapchain.h"

namespace nbl::video
{

class ILogicalDevice;

class CVKSwapchain final : public ISwapchain
{
public:
    // Todo(achal): Change order of these params
    CVKSwapchain(SCreationParams&& params, core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice)
        : ISwapchain(std::move(logicalDevice), std::move(params))
    {
#if 0
        VkSwapchainCreateInfoKHR createInfo = { VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
        createInfo.surface = static_cast<ISurfaceVK*>(m_params.surface.get())->getInternalObject();
        createInfo.minImageCount = m_params.minImageCount;
        createInfo.imageFormat = ISurfaceVK::getVkFormat(m_params.surfaceFormat.format);
        createInfo.imageColorSpace = ISurfaceVK::getVkColorSpaceKHR(m_params.surfaceFormat.colorSpace);
        createInfo.imageExtent = { m_params.width, m_params.height };
        createInfo.imageArrayLayers = m_params.arrayLayers;
        createInfo.imageUsage = static_cast<VkImageUsageFlags>(params.imageUsage);

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
#endif
    }

    ~CVKSwapchain();

    E_ACQUIRE_IMAGE_RESULT acquireNextImage(uint64_t timeout, IGPUSemaphore* semaphore, IGPUFence* fence, uint32_t* out_imgIx) override;

// Todo(achal): Remove
// private:
    VkSwapchainKHR m_vkSwapchainKHR;
};

}

#endif