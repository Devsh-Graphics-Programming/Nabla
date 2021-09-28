#include "CVulkanPipelineLayout.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanPipelineLayout::~CVulkanPipelineLayout()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipelineLayout(vulkanDevice->getInternalObject(), m_layout, nullptr);
}

}