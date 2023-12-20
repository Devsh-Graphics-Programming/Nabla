#ifndef _NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPUComputePipeline.h"

#include <volk.h>


namespace nbl::video
{
class ILogicalDevice;

class CVulkanComputePipeline final : public IGPUComputePipeline
{
    public:
        CVulkanComputePipeline(
            const ILogicalDevice* dev,
            core::smart_refctd_ptr<const IGPUShader>&& shader,
            const core::bitflag<SCreationParams::FLAGS> _flags,
            const VkPipeline pipeline
        ) : IGPUComputePipeline(core::smart_refctd_ptr<const ILogicalDevice>(dev),_flags), m_pipeline(pipeline), m_shader(std::move(shader)) {}

        inline VkPipeline getInternalObject() const { return m_pipeline; }
    
        void setObjectDebugName(const char* label) const override;

    private:
        ~CVulkanComputePipeline();

        const VkPipeline m_pipeline;
        // gotta keep that VkShaderModule alive (for now, until maintenance5)
        const core::smart_refctd_ptr<const IGPUShader> m_shader;
};
}
#endif