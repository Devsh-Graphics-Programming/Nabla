#include "nbl/video/CVulkanRenderpass.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanRenderpass::~CVulkanRenderpass()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyRenderPass(vulkanDevice->getInternalObject(), m_renderpass, nullptr);
}

}