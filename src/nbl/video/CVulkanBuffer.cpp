#include "CVulkanBuffer.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanBuffer::~CVulkanBuffer()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyBuffer(vulkanDevice->getInternalObject(), m_vkBuffer, nullptr);
}

}