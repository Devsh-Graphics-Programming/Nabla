#include "CVulkanImage.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanImage::~CVulkanImage()
{
    // auto* vk = m_vkdevice->getFunctionTable();
    // auto vkdev = m_vkdevice->getInternalObject();

    // if (this->wasCreatedBy(&vkdev))
    {
        // vk->vk.vkDestroyImage(vkdev, m_vkimg, nullptr);
        // vkDestroyImage(vkdev, m_vkimg, nullptr);
    }
}

#if 0
CVulkanImage::CVulkanImage(CVKLogicalDevice* _vkdev, IGPUImage::SCreationParams&& _params) : IGPUImage(_vkdev, std::move(_params)), m_vkdevice(_vkdev)
{
    // auto* vk = m_vkdevice->getFunctionTable();
    auto vkdev = m_vkdevice->getInternalObject();

    VkImageCreateInfo ci;
    ci.pNext = nullptr;
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.arrayLayers = params.arrayLayers;
    ci.extent = { params.extent.width, params.extent.height, params.extent.depth };
    ci.flags = static_cast<VkImageCreateFlags>(params.flags);
    ci.format = static_cast<VkFormat>(params.format);
    ci.imageType = static_cast<VkImageType>(params.type);
    ci.initialLayout = static_cast<VkImageLayout>(params.initialLayout);
    ci.mipLevels = params.mipLevels;
    ci.samples = static_cast<VkSampleCountFlagBits>(params.samples);
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.pQueueFamilyIndices = params.queueFamilyIndices->data();
    ci.queueFamilyIndexCount = params.queueFamilyIndices->size();
    ci.tiling = static_cast<VkImageTiling>(params.tiling);
    ci.usage = static_cast<VkImageUsageFlags>(params.usage);

    // vk->vk.vkCreateImage(vkdev, &ci, nullptr, &m_vkimg);
    // vkCreateImage(vkdev, &ci, nullptr, &m_vkimg);
}
#endif

}