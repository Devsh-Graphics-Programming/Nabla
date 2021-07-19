#include "nbl/video/CVulkanImageView.h"

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/CVulkanImage.h"

namespace nbl {
namespace video
{

CVulkanImageView::CVulkanImageView(CVKLogicalDevice* _vkdev, SCreationParams&& _params) : IGPUImageView(_vkdev, std::move(_params)), m_vkdev(_vkdev)
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    VkImageViewCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    ci.pNext = nullptr;
    ci.components.r = static_cast<VkComponentSwizzle>(params.components.r);
    ci.components.g = static_cast<VkComponentSwizzle>(params.components.g);
    ci.components.b = static_cast<VkComponentSwizzle>(params.components.b);
    ci.components.a = static_cast<VkComponentSwizzle>(params.components.a);
    ci.flags = static_cast<VkImageViewCreateFlags>(params.flags);
    ci.format = static_cast<VkFormat>(params.format);
    ci.subresourceRange.aspectMask = static_cast<VkImageAspectFlags>(params.subresourceRange.aspectMask);
    ci.subresourceRange.baseArrayLayer = params.subresourceRange.baseArrayLayer;
    ci.subresourceRange.baseMipLevel = params.subresourceRange.baseMipLevel;
    ci.subresourceRange.layerCount = params.subresourceRange.layerCount;
    ci.subresourceRange.levelCount = params.subresourceRange.levelCount;
    ci.viewType = static_cast<VkImageViewType>(params.viewType);
    auto vkimg = static_cast<CVulkanImage*>(params.image.get())->getInternalObject();
    ci.image = vkimg;

    vk->vk.vkCreateImageView(vkdev, &ci, nullptr, &m_vkimgview);
}

CVulkanImageView::~CVulkanImageView()
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    vk->vk.vkDestroyImageView(vkdev, m_vkimgview, nullptr);
}

}
}