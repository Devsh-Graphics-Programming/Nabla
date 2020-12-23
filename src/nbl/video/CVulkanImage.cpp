#include "CVulkanImage.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{
    CVulkanImage::~CVulkanImage()
    {
        auto* vk = m_vkdevice->getFunctionTable();
        auto vkdev = m_vkdevice->getInternalObject();

        vk->vk.vkDestroyImage(vkdev, m_vkimg, nullptr);
    }

    CVulkanImage::CVulkanImage(CVKLogicalDevice* _vkdev, IGPUImage::SCreationParams&& _params) : IGPUImage(std::move(_params)), m_vkdevice(_vkdev)
    {
        auto* vk = m_vkdevice->getFunctionTable();
        auto vkdev = m_vkdevice->getInternalObject();

        VkImageCreateInfo ci;
        ci.pNext = nullptr;
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.arrayLayers = _params.arrayLayers;
        ci.extent = { _params.extent.width, _params.extent.height, _params.extent.depth };
        ci.flags = static_cast<VkImageCreateFlags>(_params.flags);
        ci.format = static_cast<VkFormat>(_params.format);
        ci.imageType = static_cast<VkImageType>(_params.type);
        ci.initialLayout = VK_IMAGE_LAYOUT_GENERAL; // TODO
        ci.mipLevels = _params.mipLevels;
        ci.samples = static_cast<VkSampleCountFlagBits>(_params.samples);
        ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        // TODO:
        //ci.pQueueFamilyIndices = ...
        //ci.queueFamilyIndexCount = ...
        //ci.tiling = ....
        //ci.usage = ...
        vk->vk.vkCreateImage(vkdev, &ci, nullptr, &m_vkimg);
    }

}
}