#include "nbl/video/CVulkanPipelineCache.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineCache::~CVulkanPipelineCache()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipelineCache(vulkanDevice->getInternalObject(), m_pipelineCache, nullptr);
}

bool CVulkanPipelineCache::merge_impl(const std::span<const IGPUPipelineCache* const> _srcCaches)
{
    core::vector<VkPipelineCache> vk_srcCaches(_srcCaches.size());
    for (size_t i=0; i<_srcCaches.size(); i++)
        vk_srcCaches[i] = static_cast<const CVulkanPipelineCache*>(_srcCaches[i])->getInternalObject();

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    return vk->vk.vkMergePipelineCaches(vulkanDevice->getInternalObject(),m_pipelineCache,vk_srcCaches.size(),vk_srcCaches.data())==VK_SUCCESS;
}

core::smart_refctd_ptr<asset::ICPUPipelineCache> CVulkanPipelineCache::convertToCPUCache() const
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();

    size_t dataSize = 0;
    if (vk->vk.vkGetPipelineCacheData(vulkanDevice->getInternalObject(),m_pipelineCache,&dataSize,nullptr)!=VK_SUCCESS || dataSize==0ull)
        return nullptr;

    asset::ICPUPipelineCache::entries_map_t entries;
    {
        auto data = core::make_refctd_dynamic_array<core::smart_refctd_dynamic_array<uint8_t>>(dataSize);
        vk->vk.vkGetPipelineCacheData(vulkanDevice->getInternalObject(),m_pipelineCache,&dataSize,data->data());
        entries[vulkanDevice->getPipelineCacheKey()] = {std::move(data)};
    }
    return core::make_smart_refctd_ptr<asset::ICPUPipelineCache>(std::move(entries));
}

void CVulkanPipelineCache::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(vkSetDebugUtilsObjectNameEXT == 0) return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_PIPELINE_CACHE;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

}