#ifndef __NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED__

#include <volk.h>

namespace nbl::video
{

class CVulkanComputePipeline : public IGPUComputePipeline
{

private:
    VkPipeline m_pipeline;
};

}

#define __NBL_VIDEO_C_VULKAN_COMPUTE_PIPELINE_H_INCLUDED__
#endif
