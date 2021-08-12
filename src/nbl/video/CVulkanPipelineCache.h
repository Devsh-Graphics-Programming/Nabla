#ifndef __NBL_VIDEO_C_VULKAN_PIPELINE_CACHE_H_INCLUDED__

#include "nbl/video/IGPUPipelineCache.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanPipelineCache : public IGPUPipelineCache
{
public:
    CVulkanPipelineCache(ILogicalDevice* dev, VkPipelineCache pipelineCache)
        : IGPUPipelineCache(dev), m_pipelineCache(pipelineCache)
    {}

    ~CVulkanPipelineCache();

    inline VkPipelineCache getInternalObject() const { return m_pipelineCache; }

private:
    VkPipelineCache m_pipelineCache;
};

}

#define __NBL_VIDEO_C_VULKAN_PIPELINE_CACHE_H_INCLUDED__
#endif
