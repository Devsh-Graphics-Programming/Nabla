#include "CVulkanSampler.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanSampler::~CVulkanSampler()
{
    VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
    vkDestroySampler(vk_device, m_sampler, nullptr);
}

}