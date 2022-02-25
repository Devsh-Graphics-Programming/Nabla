#ifndef __NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED__

#include "nbl/video/IGPUComputePipeline.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanComputePipeline : public IGPUComputePipeline
{
public:
    CVulkanComputePipeline(core::smart_refctd_ptr<ILogicalDevice>&& dev,
        core::smart_refctd_ptr<IGPUPipelineLayout>&& layout, 
        core::smart_refctd_ptr<IGPUSpecializedShader>&& shader, VkPipeline pipeline)
        : IGPUComputePipeline(std::move(dev), std::move(layout), std::move(shader)), m_pipeline(pipeline)
    {}

    ~CVulkanComputePipeline();

    inline VkPipeline getInternalObject() const { return m_pipeline; }
    
    void setObjectDebugName(const char* label) const override;

private:
    VkPipeline m_pipeline;
};

}

#define __NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED__
#endif
