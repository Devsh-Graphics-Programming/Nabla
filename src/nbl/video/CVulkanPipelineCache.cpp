#include "CVulkanPipelineCache.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineCache::~CVulkanPipelineCache()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyPipelineCache(device, m_pipelineCache, nullptr);
    }
}

}