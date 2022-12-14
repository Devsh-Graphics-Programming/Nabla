#include "nbl/video/CVulkanQueue.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanFence.h"
#include "nbl/video/CVulkanSemaphore.h"
#include "nbl/video/CVulkanCommandBuffer.h"

namespace nbl::video
{

bool CVulkanQueue::submit(uint32_t _count, const SSubmitInfo* _submits, IGPUFence* _fence)
{
    if (!IGPUQueue::submit(_count, _submits, _fence))
        return false;

    if(!IGPUQueue::markCommandBuffersAsPending(_count, _submits))
        return false;

    auto* vk = static_cast<const CVulkanLogicalDevice*>(m_originDevice)->getFunctionTable();

    uint32_t waitSemCnt = 0u;
    uint32_t signalSemCnt = 0u;
    uint32_t cmdBufCnt = 0u;

    for (uint32_t i = 0u; i < _count; ++i)
    {
        const auto& sb = _submits[i];
        waitSemCnt += sb.waitSemaphoreCount;
        signalSemCnt += sb.signalSemaphoreCount;
        cmdBufCnt += sb.commandBufferCount;
    }

    constexpr uint32_t STACK_MEM_SIZE = 1u<<15;
    uint8_t stackmem_[STACK_MEM_SIZE]{};
    uint8_t* mem = stackmem_;
    uint32_t memSize = STACK_MEM_SIZE;

    const uint32_t submitsSz = sizeof(VkSubmitInfo)*_count;
    const uint32_t memNeeded = submitsSz + (waitSemCnt + signalSemCnt)*sizeof(VkSemaphore) + cmdBufCnt*sizeof(VkCommandBuffer);
    if (memNeeded > memSize)
    {
        memSize = memNeeded;
        mem = reinterpret_cast<uint8_t*>( _NBL_ALIGNED_MALLOC(memSize, _NBL_SIMD_ALIGNMENT) );
    }

    // Todo(achal): FREE ONLY IF _NBL_ALIGNED_MALLOC was called
    // auto raii_ = core::makeRAIIExiter([mem]{ _NBL_ALIGNED_FREE(mem); });

    VkSubmitInfo* submits = reinterpret_cast<VkSubmitInfo*>(mem);
    mem += submitsSz;

    VkSemaphore* waitSemaphores = reinterpret_cast<VkSemaphore*>(mem);
    mem += waitSemCnt*sizeof(VkSemaphore);
    VkSemaphore* signalSemaphores = reinterpret_cast<VkSemaphore*>(mem);
    mem += signalSemCnt*sizeof(VkSemaphore);
    VkCommandBuffer* cmdbufs = reinterpret_cast<VkCommandBuffer*>(mem);
    mem += cmdBufCnt*sizeof(VkCommandBuffer);

    uint32_t waitSemOffset = 0u;
    uint32_t signalSemOffset = 0u;
    uint32_t cmdBufOffset = 0u;
    for (uint32_t i = 0u; i < _count; ++i)
    {
        auto& sb = submits[i];

        const SSubmitInfo& _sb = _submits[i];
#ifdef _NBL_DEBUG
        for (uint32_t j = 0u; j < _sb.commandBufferCount; ++j)
            assert(_sb.commandBuffers[j]->getLevel() != CVulkanCommandBuffer::EL_SECONDARY);
#endif

        sb.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        sb.pNext = nullptr;
        VkCommandBuffer* commandBuffers = cmdbufs + cmdBufOffset;
        sb.pCommandBuffers = commandBuffers;
        sb.commandBufferCount = _sb.commandBufferCount;
        auto* waits = waitSemaphores + waitSemOffset;
        sb.pWaitSemaphores = waits;
        sb.waitSemaphoreCount = _sb.waitSemaphoreCount;
        auto* signals = signalSemaphores + signalSemOffset;
        sb.pSignalSemaphores = signals;
        sb.signalSemaphoreCount = _sb.signalSemaphoreCount;

        for (uint32_t j = 0u; j < sb.waitSemaphoreCount; ++j)
        {
            waits[j] = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(_sb.pWaitSemaphores[j], m_originDevice)->getInternalObject();
        }
        for (uint32_t j = 0u; j < sb.signalSemaphoreCount; ++j)
        {
            signals[j] = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(_sb.pSignalSemaphores[j], m_originDevice)->getInternalObject();
        }
        for (uint32_t j = 0u; j < sb.commandBufferCount; ++j)
        {
            commandBuffers[j] = reinterpret_cast<CVulkanCommandBuffer*>(_sb.commandBuffers[j])->getInternalObject();
        }

        waitSemOffset += sb.waitSemaphoreCount;
        signalSemOffset += sb.signalSemaphoreCount;
        cmdBufOffset += sb.commandBufferCount;

        static_assert(sizeof(VkPipelineStageFlags) == sizeof(asset::E_PIPELINE_STAGE_FLAGS));
        sb.pWaitDstStageMask = reinterpret_cast<const VkPipelineStageFlags*>(_sb.pWaitDstStageMask);
    }

    VkFence fence = _fence ? IBackendObject::device_compatibility_cast<CVulkanFence*>(_fence, m_originDevice)->getInternalObject() : VK_NULL_HANDLE;
    if (vk->vk.vkQueueSubmit(m_vkQueue, _count, submits, fence) == VK_SUCCESS)
    {
        if(!IGPUQueue::markCommandBuffersAsDone(_count, _submits))
            return false;

        return true;
    }

    return false;
}

bool CVulkanQueue::startCapture() 
{
	if(m_rdoc_api == nullptr)
		return false;
    m_rdoc_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(m_vkInstance), NULL);
	return true;
}
bool CVulkanQueue::endCapture()
{
	if(m_rdoc_api == nullptr)
		return false;
    m_rdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(m_vkInstance), NULL);
	return true;
}
}