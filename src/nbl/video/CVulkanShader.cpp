#include "CVulkanShader.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{
CVulkanShader::~CVulkanShader()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyShaderModule(vulkanDevice->getInternalObject(), m_vkShaderModule, nullptr);
}

}