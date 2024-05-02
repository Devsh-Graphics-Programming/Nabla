#ifndef _NBL_VIDEO_C_VULKAN_PIPELINE_LAYOUT_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_PIPELINE_LAYOUT_H_INCLUDED_

#include "nbl/video/IGPUPipelineLayout.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanPipelineLayout : public IGPUPipelineLayout
{
    public:
        CVulkanPipelineLayout(
            const ILogicalDevice* dev, const std::span<const asset::SPushConstantRange> _pcRanges,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
            core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2, core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3,
            const VkPipelineLayout vk_layout
        ) : IGPUPipelineLayout(
                core::smart_refctd_ptr<const ILogicalDevice>(dev),
                _pcRanges,std::move(_layout0),std::move(_layout1),std::move(_layout2),std::move(_layout3)
            ), m_layout(vk_layout) {}

        ~CVulkanPipelineLayout();

        inline VkPipelineLayout getInternalObject() const { return m_layout; }

        void setObjectDebugName(const char* label) const override;

    private:
        const VkPipelineLayout m_layout;

};

}
#endif