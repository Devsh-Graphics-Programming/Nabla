#include "CVulkanDescriptorSetLayout.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanDescriptorSetLayout::~CVulkanDescriptorSetLayout()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyDescriptorSetLayout(vulkanDevice->getInternalObject(), m_dsLayout, nullptr);
}

}