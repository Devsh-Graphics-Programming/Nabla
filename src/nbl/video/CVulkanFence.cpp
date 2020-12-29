#include "nbl/video/CVulkanFence.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

CVulkanFence::CVulkanFence(CVKLogicalDevice* _vkdev, E_CREATE_FLAGS _flags) : IGPUFence(_flags), m_vkdev(_vkdev)
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    VkFenceCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    ci.pNext = nullptr;
    ci.flags = static_cast<VkFenceCreateFlags>(_flags);
    vk->vk.vkCreateFence(vkdev, &ci, nullptr, &m_fence);
}

CVulkanFence::~CVulkanFence()
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    vk->vk.vkDestroyFence(vkdev, m_fence, nullptr);
}

}
}