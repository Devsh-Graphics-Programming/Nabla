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

bool CVulkanDescriptorPool::createDescriptorSets_impl(uint32_t count, const IGPUDescriptorSetLayout* const* layouts, SDescriptorOffsets* const offsets, const uint32_t firstSetOffsetInPool, core::smart_refctd_ptr<IGPUDescriptorSet>* output)
{
    VkDescriptorSetAllocateInfo vk_allocateInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    vk_allocateInfo.pNext = nullptr; // pNext must be NULL or a pointer to a valid instance of VkDescriptorSetVariableDescriptorCountAllocateInfo

    vk_allocateInfo.descriptorPool = m_descriptorPool;
    vk_allocateInfo.descriptorSetCount = count;

    core::vector<VkDescriptorSetLayout> vk_dsLayouts(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        assert(layouts[i]->getAPIType() == EAT_VULKAN);
        vk_dsLayouts[i] = IBackendObject::device_compatibility_cast<const CVulkanDescriptorSetLayout*>(layouts[i], getOriginDevice())->getInternalObject();
    }

    vk_allocateInfo.pSetLayouts = vk_dsLayouts.data();

    core::vector<VkDescriptorSet> vk_descriptorSets(count);

    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    if (vk->vk.vkAllocateDescriptorSets(vulkanDevice->getInternalObject(), &vk_allocateInfo, vk_descriptorSets.data()) == VK_SUCCESS)
    {
        for (uint32_t i = 0; i < count; ++i)
            output[i] = core::make_smart_refctd_ptr<CVulkanDescriptorSet>(core::smart_refctd_ptr<const IGPUDescriptorSetLayout>(layouts[i]), core::smart_refctd_ptr<IDescriptorPool>(this), firstSetOffsetInPool + i, std::move(offsets[i]), vk_descriptorSets[i]);

        return true;
    }

    return false;
}

bool CVulkanDescriptorPool::reset_impl()
{
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    const bool success = (vk->vk.vkResetDescriptorPool(vulkanDevice->getInternalObject(), m_descriptorPool, 0) == VK_SUCCESS);
    return success;
}

}