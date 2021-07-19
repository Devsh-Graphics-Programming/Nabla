#include "nbl/video/CVulkanFence.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl
{
namespace video
{

CVulkanFence::CVulkanFence(CVKLogicalDevice* _vkdev, E_CREATE_FLAGS _flags) : IGPUFence(_vkdev, _flags), m_vkdev(_vkdev)
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    VkFenceCreateInfo createInfo = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    createInfo.pNext = nullptr;
    createInfo.flags = static_cast<VkFenceCreateFlags>(_flags);
    vk->vk.vkCreateFence(vkdev, &createInfo, nullptr, &m_fence);
}

CVulkanFence::~CVulkanFence()
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    vk->vk.vkDestroyFence(vkdev, m_fence, nullptr);
}

}
}