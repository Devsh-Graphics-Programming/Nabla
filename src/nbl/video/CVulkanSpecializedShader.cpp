#include "CVulkanSpecializedShader.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanSpecializedShader::~CVulkanSpecializedShader()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyShaderModule(vulkanDevice->getInternalObject(), m_shaderModule, nullptr);
}

}