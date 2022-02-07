#include "CVulkanDescriptorPool.h"

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

    if(vkSetDebugUtilsObjectNameEXT == 0)
        return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
    nameInfo.objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL;
    nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
    nameInfo.pObjectName = getObjectDebugName();
    vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}
}