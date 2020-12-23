#include "nbl/video/CVulkanQueue.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

CVulkanQueue::CVulkanQueue(CVKLogicalDevice* vkdev, VkQueue vkq, uint32_t _famIx, E_CREATE_FLAGS _flags, float _priority) : 
    IGPUQueue(_famIx, _flags, _priority), m_vkdevice(vkdev), m_vkqueue(vkq)
{
    
}

void CVulkanQueue::submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence)
{
    auto vkdev = m_vkdevice->getInternalObject();
    auto* vk = m_vkdevice->getFunctionTable();

    core::vector<VkSubmitInfo> info_heap;
    constexpr size_t MAX_SUBMITS_ON_STACK = 50u;
    VkSubmitInfo info_stack[MAX_SUBMITS_ON_STACK];

    VkSubmitInfo* submits = info_stack;
    if (_count > MAX_SUBMITS_ON_STACK)
    {
        info_heap.resize(_count);
        submits = info_heap.data();
    }

    for (uint32_t i = 0u; i < _count; ++i)
    {
        auto& sb = submits[i];
        auto& _sb = _submits[i];

        sb.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        sb.pNext = nullptr;
        sb.commandBufferCount = _sb.commandBufferCount;
        // TODO get semaphores and cmd bufs vk handles
    }

    VkFence fence; // TODO get fence vk handle
    vk->vk.vkQueueSubmit(m_vkqueue, _count, info, fence);
}

}
}