#ifndef _NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED_


#include "nbl/video/IGPUComputePipeline.h"

#include "nbl/video/CVulkanShader.h"

#include <volk.h>


namespace nbl::video
{
class ILogicalDevice;

class CVulkanComputePipeline final : public IGPUComputePipeline
{
    public:
        CVulkanComputePipeline(
            core::smart_refctd_ptr<const IGPUPipelineLayout>&& _layout,
            core::smart_refctd_ptr<const CVulkanShader>&& _shader,
            const core::bitflag<SCreationParams::FLAGS> _flags,
            const VkPipeline pipeline
        ) : IGPUComputePipeline(std::move(_layout),_flags),
            m_pipeline(pipeline), m_shader(std::move(_shader)) {}

        inline const void* getNativeHandle() const override { return &m_pipeline; }

        inline VkPipeline getInternalObject() const { return m_pipeline; }
    
        void setObjectDebugName(const char* label) const override;

    private:
        ~CVulkanComputePipeline();

        const VkPipeline m_pipeline;
        // gotta keep that VkShaderModule alive (for now, until maintenance5)
        const core::smart_refctd_ptr<const CVulkanShader> m_shader;
};
}
#endif