#ifndef __NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED__
#define __NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED__

#include <volk.h>

#include "nbl/video/IGPUImageView.h"

namespace nbl::video
{

class CVKLogicalDevice;

class CVulkanImageView final : public IGPUImageView
{
public:
    CVulkanImageView(CVKLogicalDevice* _vkdev, SCreationParams&& _params);
    ~CVulkanImageView();

    inline VkImageView getInternalObject() const { return m_vkimgview; }

private:
    CVKLogicalDevice* m_vkdev;
    VkImageView m_vkimgview;
};

}

#endif
