#ifndef __NBL_VIDEO_C_VULKAN_PIPELINE_LAYOUT_H_INCLUDED__

#include "nbl/video/IGPUPipelineLayout.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanPipelineLayout : public IGPUPipelineLayout
{
public:
    CVulkanPipelineLayout(core::smart_refctd_ptr<ILogicalDevice>&& dev,
        const asset::SPushConstantRange* const _pcRangesBegin,
        const asset::SPushConstantRange* const _pcRangesEnd,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout0,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout1,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout2,
        core::smart_refctd_ptr<IGPUDescriptorSetLayout>&& _layout3, VkPipelineLayout vk_layout)
        : IGPUPipelineLayout(std::move(dev), _pcRangesBegin, _pcRangesEnd, std::move(_layout0),
            std::move(_layout1), std::move(_layout2), std::move(_layout3)), m_layout(vk_layout)
    {}

    ~CVulkanPipelineLayout();

    inline VkPipelineLayout getInternalObject() const { return m_layout; }

private:
    VkPipelineLayout m_layout;

};

}

#define __NBL_VIDEO_C_VULKAN_PIPELINE_LAYOUT_H_INCLUDED__
#endif