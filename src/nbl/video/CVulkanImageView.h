#ifndef __NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED__
#define __NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED__

#include <volk.h>

#include "nbl/video/IGPUImageView.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanImageView final : public IGPUImageView
{
public:
    CVulkanImageView(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
        SCreationParams&& _params) : IGPUImageView(std::move(logicalDevice), std::move(_params))
    {
#if 0
        // auto* vk = m_vkdev->getFunctionTable();
        auto vkdev = m_vkdev->getInternalObject();

        VkImageViewCreateInfo createInfo = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        createInfo.pNext = nullptr;
        createInfo.flags = static_cast<VkImageViewCreateFlags>(params.flags);

        auto vkimg = static_cast<CVulkanImage*>(params.image.get())->getInternalObject();
        createInfo.image = vkimg;
        createInfo.viewType = static_cast<VkImageViewType>(params.viewType);
        createInfo.format = ISurfaceVK::getVkFormat(params.format);
        createInfo.components.r = static_cast<VkComponentSwizzle>(params.components.r);
        createInfo.components.g = static_cast<VkComponentSwizzle>(params.components.g);
        createInfo.components.b = static_cast<VkComponentSwizzle>(params.components.b);
        createInfo.components.a = static_cast<VkComponentSwizzle>(params.components.a);
        createInfo.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(params.subresourceRange.aspectMask);
        createInfo.subresourceRange.baseMipLevel = params.subresourceRange.baseMipLevel;
        createInfo.subresourceRange.levelCount = params.subresourceRange.levelCount;
        createInfo.subresourceRange.baseArrayLayer = params.subresourceRange.baseArrayLayer;
        createInfo.subresourceRange.layerCount = params.subresourceRange.layerCount;

        // vk->vk.vkCreateImageView(vkdev, &createInfo, nullptr, &m_vkimgview);
        vkCreateImageView(vkdev, &createInfo, nullptr, &m_vkimgview);
#endif
    }

    ~CVulkanImageView();

    inline VkImageView getInternalObject() const { return m_vkImageView; }

private:
    VkImageView m_vkImageView;
};

}

#endif
