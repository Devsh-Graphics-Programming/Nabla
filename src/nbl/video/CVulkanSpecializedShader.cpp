#include "CVulkanSpecializedShader.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanSpecializedShader::~CVulkanSpecializedShader()
{
    auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = reinterpret_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyShaderModule(device, m_shaderModule, nullptr);
    }

}

}