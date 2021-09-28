#include "CVulkanPipelineCache.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineCache::~CVulkanPipelineCache()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipelineCache(vulkanDevice->getInternalObject(), m_pipelineCache, nullptr);
}

}