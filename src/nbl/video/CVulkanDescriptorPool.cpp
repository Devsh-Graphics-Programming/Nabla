#include "nbl/video/CVulkanDescriptorPool.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorPool::~CVulkanDescriptorPool()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyDescriptorPool(vulkanDevice->getInternalObject(), m_descriptorPool, nullptr);
}

void CVulkanDescriptorPool::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if(vkSetDebugUtilsObjectNameEXT == 0) return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

bool CVulkanDescriptorPool::freeDescriptorSets_impl(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets)
{
    constexpr auto MaxDescriptorSetCount = 4u;
    assert(descriptorSetCount <= MaxDescriptorSetCount);
    VkDescriptorSet vk_descriptorSets[MaxDescriptorSetCount];

    for (auto i = 0; i < descriptorSetCount; ++i)
    {
        if (descriptorSets[i]->getAPIType() != EAT_VULKAN)
            return false;

        vk_descriptorSets[i] = static_cast<CVulkanDescriptorSet*>(descriptorSets[i])->getInternalObject();
    }

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    return vk->vk.vkFreeDescriptorSets(vulkanDevice->getInternalObject(), m_descriptorPool, descriptorSetCount, vk_descriptorSets) == VK_SUCCESS;
}

}