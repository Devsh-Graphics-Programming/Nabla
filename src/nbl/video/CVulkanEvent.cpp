#include "CVulkanEvent.h"

#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanEvent::~CVulkanEvent()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyEvent(vulkanDevice->getInternalObject(), m_vkEvent, nullptr);
}

}