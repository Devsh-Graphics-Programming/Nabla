#ifndef _NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_DESCRIPTOR_SET_LAYOUT_H_INCLUDED_


#include "nbl/video/IGPUDescriptorSetLayout.h"


namespace nbl::video
{

class CVulkanDescriptorSetLayout : public IGPUDescriptorSetLayout
{
    public:
        CVulkanDescriptorSetLayout(const ILogicalDevice* dev, const std::span<const SBinding> _bindings, VkDescriptorSetLayout vk_dsLayout)
            : IGPUDescriptorSetLayout(core::smart_refctd_ptr<const ILogicalDevice>(dev),_bindings), m_dsLayout(vk_dsLayout) {}

        ~CVulkanDescriptorSetLayout();

        inline VkDescriptorSetLayout getInternalObject() const { return m_dsLayout; }

        void setObjectDebugName(const char* label) const override;

    private:
        const VkDescriptorSetLayout m_dsLayout;
};

}
#endif
