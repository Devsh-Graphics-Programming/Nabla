#ifndef __NBL_VIDEO_C_VULKAN_SAMPLER_H_INCLUDED__

#include "nbl/video/IGPUSampler.h"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

namespace nbl::video
{

class CVulkanSampler : public IGPUSampler
{
public:
    CVulkanSampler(core::smart_refctd_ptr<const ILogicalDevice>&& dev, const SParams& params,
        const VkSampler vk_sampler)
        : IGPUSampler(std::move(dev), params), m_sampler(vk_sampler)
    {}

    ~CVulkanSampler();

    const void* getNativeHandle() const override {return &m_sampler;}
    inline VkSampler getInternalObject() const {return m_sampler;}

    void setObjectDebugName(const char* label) const override;

private:
    VkSampler m_sampler;
};

}

#define __NBL_VIDEO_C_VULKAN_SAMPLER_H_INCLUDED__
#endif
