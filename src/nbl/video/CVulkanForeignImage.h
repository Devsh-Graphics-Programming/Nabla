#ifndef __NBL_VIDEO_C_VULKAN_FOREIGN_IMAGE_H_INCLUDED__

#include "nbl/video/CVulkanImage.h"

namespace nbl::video
{

class CVulkanSwapchain;

class CVulkanForeignImage final : public CVulkanImage
{
public:
    CVulkanForeignImage(core::smart_refctd_ptr<ILogicalDevice>&& _vkdev,
        IGPUImage::SCreationParams&& _params, VkImage _vkimg)
        : CVulkanImage(std::move(_vkdev), std::move(_params), _vkimg, IDriverMemoryBacked::SDriverMemoryRequirements2{/*TODO(Erfan):should any reqs be specified for foreign image???*/})
    {}

    ~CVulkanForeignImage();

private:
    // circular dep
    // core::smart_refctd_ptr<CVulkanSwapchain> m_swapchain; // the only foreigner we have now is swapchain
};

}

#define __NBL_VIDEO_C_VULKAN_FOREIGN_IMAGE_H_INCLUDED__
#endif