#ifndef __NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED__
#define __NBL_C_VULKAN_IMAGE_VIEW_H_INCLUDED__

#include "nbl/video/IGPUImageView.h"

#define VK_NO_PROTOTYPES
#include "vulkan/vulkan.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanImageView final : public IGPUImageView
{
public:
    CVulkanImageView(core::smart_refctd_ptr<ILogicalDevice>&& logicalDevice,
        SCreationParams&& _params, VkImageView imageView)
        : IGPUImageView(std::move(logicalDevice), std::move(_params)), m_vkImageView(imageView)
    {}

    ~CVulkanImageView();

    inline VkImageView getInternalObject() const { return m_vkImageView; }

private:
    VkImageView m_vkImageView;
};

}

#endif
