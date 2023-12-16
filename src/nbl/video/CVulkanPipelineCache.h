#ifndef _NBL_VIDEO_C_VULKAN_PIPELINE_CACHE_H_INCLUDED_
#define _NBL_VIDEO_C_VULKAN_PIPELINE_CACHE_H_INCLUDED_


#include "nbl/video/IGPUPipelineCache.h"

#include <volk.h>


namespace nbl::video
{
class ILogicalDevice;

class CVulkanPipelineCache final : public IGPUPipelineCache
{
    public:
        CVulkanPipelineCache(core::smart_refctd_ptr<ILogicalDevice>&& dev, const VkPipelineCache pipelineCache)
            : IGPUPipelineCache(std::move(dev)), m_pipelineCache(pipelineCache) {}

        inline VkPipelineCache getInternalObject() const {return m_pipelineCache;}

        void setObjectDebugName(const char* label) const override;

    private:
        ~CVulkanPipelineCache();

        const VkPipelineCache m_pipelineCache;
};
}
#endif
