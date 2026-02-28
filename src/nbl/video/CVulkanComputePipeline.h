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
            const SCreationParams& params,
            const VkPipeline pipeline
        ) : IGPUComputePipeline(params), m_pipeline(pipeline) {}

        inline const void* getNativeHandle() const override { return &m_pipeline; }

        inline VkPipeline getInternalObject() const { return m_pipeline; }

        void populateExecutableInfo(bool includeInternalRepresentations);

        void setObjectDebugName(const char* label) const override;

    private:
        ~CVulkanComputePipeline();

        const VkPipeline m_pipeline;
};
}
#endif