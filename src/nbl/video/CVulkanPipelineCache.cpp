#include "CVulkanPipelineCache.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineCache::~CVulkanPipelineCache()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVulkanLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyPipelineCache(device, m_pipelineCache, nullptr);
    }
}

}