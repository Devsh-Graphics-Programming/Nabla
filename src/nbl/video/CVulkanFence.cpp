#include "nbl/video/CVulkanFence.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanFence::~CVulkanFence()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyFence(vulkanDevice->getInternalObject(), m_fence, nullptr);
}

}