#ifndef __NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__

#include "nbl/video/IGPUDescriptorSetLayout.h"

namespace nbl::video
{

class ILogicalDevice;

class CVulkanDescriptorSetLayout : public IGPUDescriptorSetLayout
{
public:
    CVulkanDescriptorSetLayout(ILogicalDevice* dev, const SBinding* const _begin,
        const SBinding* const _end, VkDescriptorSetLayout vk_dsLayout)
        : IGPUDescriptorSetLayout(dev, _begin, _end), m_dsLayout(vk_dsLayout)
    {}

    ~CVulkanDescriptorSetLayout();

    inline VkDescriptorSetLayout getInternalObject() const { return m_dsLayout; }

private:
    VkDescriptorSetLayout m_dsLayout;
};

}

#define __NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_LAYOUT_H_INCLUDED__
#endif
