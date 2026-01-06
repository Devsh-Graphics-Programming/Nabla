#include "nbl/video/CVulkanGraphicsPipeline.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "CVulkanPipelineExecutableInfo.h"

namespace nbl::video
{

void CVulkanGraphicsPipeline::populateExecutableInfo(bool includeInternalRepresentations)
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    populateExecutableInfoFromVulkan(m_executableInfo, vulkanDevice->getFunctionTable(), vulkanDevice->getInternalObject(), m_vkPipeline, includeInternalRepresentations);
}

CVulkanGraphicsPipeline::~CVulkanGraphicsPipeline()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyPipeline(vulkanDevice->getInternalObject(), m_vkPipeline, nullptr);
}

}