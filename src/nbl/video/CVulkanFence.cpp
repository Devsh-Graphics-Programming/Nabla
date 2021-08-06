#include "nbl/video/CVulkanFence.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{

CVulkanFence::CVulkanFence(CVKLogicalDevice* _vkdev, E_CREATE_FLAGS _flags, VkFence fence)
    : IGPUFence(_vkdev, _flags), m_vkdev(_vkdev), m_fence(fence)
{
}

CVulkanFence::~CVulkanFence()
{
    // auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    // vk->vk.vkDestroyFence(vkdev, m_fence, nullptr);
    vkDestroyFence(vkdev, m_fence, nullptr);
}

}