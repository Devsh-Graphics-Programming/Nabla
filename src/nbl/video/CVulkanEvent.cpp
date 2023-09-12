#include "nbl/video/CVulkanEvent.h"
#include "nbl/video/CVulkanLogicalDevice.h"


namespace nbl::video
{


CVulkanEvent::~CVulkanEvent()
{
    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    auto* vk = vulkanDevice->getFunctionTable();
    vk->vk.vkDestroyEvent(vulkanDevice->getInternalObject(), m_vkEvent, nullptr);
}


auto CVulkanEvent::getEventStatus_impl() const -> STATUS
{
    const auto vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    switch (vulkanDevice->getFunctionTable()->vk.vkGetEventStatus(vulkanDevice->getInternalObject(),m_vkEvent))
    {
        case VK_EVENT_SET:
            return IEvent::STATUS::SET;
        case VK_EVENT_RESET:
            return IEvent::STATUS::RESET;
        default:
            break;
    }
    return IEvent::STATUS::FAILURE;
}

auto CVulkanEvent::resetEvent_impl() -> STATUS
{
    const auto vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    if (vulkanDevice->getFunctionTable()->vk.vkResetEvent(vulkanDevice->getInternalObject(),m_vkEvent)==VK_SUCCESS)
        return IEvent::STATUS::RESET;
    else
        return IEvent::STATUS::FAILURE;
}

auto CVulkanEvent::setEvent_impl() -> STATUS
{
    const auto vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    if (vulkanDevice->getFunctionTable()->vk.vkSetEvent(vulkanDevice->getInternalObject(),m_vkEvent)==VK_SUCCESS)
        return IEvent::STATUS::SET;
    else
        return IEvent::STATUS::FAILURE;
}

}