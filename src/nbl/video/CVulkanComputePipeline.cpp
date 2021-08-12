#include "CVulkanComputePipeline.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanComputePipeline::~CVulkanComputePipeline()
{
    const auto originDevice = getOriginDevice();
    if (originDevice->getAPIType() == EAT_VULKAN)
    {
        VkDevice device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
        vkDestroyPipeline(device, m_pipeline, nullptr);
    }
}

}