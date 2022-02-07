#include "CVulkanGraphicsPipeline.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
CVulkanGraphicsPipeline::~CVulkanGraphicsPipeline()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipeline(vulkanDevice->getInternalObject(), m_vkPipeline, nullptr);
}

}