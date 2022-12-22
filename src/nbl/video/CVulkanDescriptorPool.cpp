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

core::smart_refctd_ptr<IGPUDescriptorSet> CVulkanDescriptorPool::createDescriptorSet_impl(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>&& layout, SDescriptorOffsets&& offsets)
{
    if (layout->getAPIType() != EAT_VULKAN)
        return nullptr;

    VkDescriptorSetAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    vk_allocateInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDescriptorSetVariableDescriptorCountAllocateInfo

    vk_allocateInfo.descriptorPool = m_descriptorPool;
    vk_allocateInfo.descriptorSetCount = 1u;

    VkDescriptorSetLayout vk_dsLayout = IBackendObject::device_compatibility_cast<const CVulkanDescriptorSetLayout*>(layout.get(), getOriginDevice())->getInternalObject();
    vk_allocateInfo.pSetLayouts = &vk_dsLayout;

    VkDescriptorSet vk_descriptorSet;

    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    if (vk->vk.vkAllocateDescriptorSets(vulkanDevice->getInternalObject(), &vk_allocateInfo, &vk_descriptorSet) == VK_SUCCESS)
        return core::make_smart_refctd_ptr<CVulkanDescriptorSet>(std::move(layout), core::smart_refctd_ptr<IDescriptorPool>(this), std::move(offsets), vk_descriptorSet);

    return nullptr;
}

bool CVulkanDescriptorPool::freeDescriptorSets_impl(const uint32_t descriptorSetCount, IGPUDescriptorSet* const* const descriptorSets)
{
    core::vector<VkDescriptorSet> vk_descriptorSets(descriptorSetCount, VK_NULL_HANDLE);

    for (auto i = 0; i < descriptorSetCount; ++i)
    {
        if (descriptorSets[i]->getAPIType() != EAT_VULKAN)
            return false;

        vk_descriptorSets[i] = static_cast<CVulkanDescriptorSet*>(descriptorSets[i])->getInternalObject();
    }

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    return vk->vk.vkFreeDescriptorSets(vulkanDevice->getInternalObject(), m_descriptorPool, descriptorSetCount, vk_descriptorSets.data()) == VK_SUCCESS;
}

}