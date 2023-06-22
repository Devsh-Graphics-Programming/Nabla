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

    // TODO: we need a SVO vector
    core::vector<VkSubmitInfo2> submits(_count,{VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,/*No interesting extensions*/nullptr,/*No protected stuff yet*/0});
    core::vector<VkSemaphoreSubmitInfoKHR> waitSemaphores(waitSemCnt,{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,nullptr});
    core::vector<VkCommandBufferSubmitInfoKHR> commandBuffers(cmdBufCnt,{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,nullptr});
    core::vector<VkSemaphoreSubmitInfoKHR> signalSemaphores(waitSemCnt,{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR,nullptr});

    uint32_t waitSemOffset = 0u;
    uint32_t signalSemOffset = 0u;
    uint32_t cmdBufOffset = 0u;
    for (uint32_t i=0u; i<_count; ++i)
    {
        auto& sb = submits[i];

        const SSubmitInfo& _sb = _submits[i];
        #ifdef _NBL_DEBUG
        // TODO: timeline semaphores https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubmitInfo2-semaphore-03881
        // TODO: timeline semaphores https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubmitInfo2-semaphore-03882
        // TODO: timeline semaphores https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubmitInfo2-semaphore-03883
        // TODO: timeline semaphores https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#VUID-VkSubmitInfo2-semaphore-03884
        for (uint32_t j=0u; j<_sb.commandBufferCount; ++j)
        {
            assert(_sb.commandBuffers[j]->getLevel()!=CVulkanCommandBuffer::EL_SECONDARY);
        }
        #endif

        auto* waits = waitSemaphores.data()+waitSemOffset;
        sb.pWaitSemaphoreInfos = waits;
        sb.waitSemaphoreInfoCount = _sb.waitSemaphoreCount;
        auto* cmdbufs = commandBuffers.data()+cmdBufOffset;
        sb.pCommandBufferInfos = cmdbufs;
        sb.commandBufferInfoCount = _sb.commandBufferCount;
        auto* signals = signalSemaphores.data()+signalSemOffset;
        sb.pSignalSemaphoreInfos = signals;
        sb.signalSemaphoreInfoCount = _sb.signalSemaphoreCount;

        for (uint32_t j=0u; j<_sb.waitSemaphoreCount; ++j)
        {
            waits[j].semaphore = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(_sb.pWaitSemaphores[j], m_originDevice)->getInternalObject();
            waits[j].value = 0xdeafbeefu; // ignored, because timeline semaphores are a TODO
            waits[j].stageMask = ;
            waits[j].deviceIndex = 0u; // device groups are a TODO
        }
        for (uint32_t j=0u; j<_sb.signalSemaphoreCount; ++j)
        {
            signals[j] = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(_sb.pSignalSemaphores[j], m_originDevice)->getInternalObject();
        }
        for (uint32_t j=0u; j<_sb.commandBufferCount; ++j)
        {
            commandBuffers[j] = reinterpret_cast<CVulkanCommandBuffer*>(_sb.commandBuffers[j])->getInternalObject();
        }

        waitSemOffset += _sb.waitSemaphoreCount;
        signalSemOffset += _sb.signalSemaphoreCount;
        cmdBufOffset += _sb.commandBufferCount;

        sb.pWaitDstStageMask = getVkPipelineStageFlagsFromPipelineStageFlags(_sb.pWaitDstStageMask);
    }

    VkFence fence = _fence ? IBackendObject::device_compatibility_cast<CVulkanFence*>(_fence, m_originDevice)->getInternalObject() : VK_NULL_HANDLE;
    auto vkRes = vk->vk.vkQueueSubmit2KHR(m_vkQueue, _count, submits.data(), fence);
    if (vkRes == VK_SUCCESS)
    {
        if(!IGPUQueue::markCommandBuffersAsDone(_count, _submits))
            return false;

        return true;
    }
    else
    {
        _NBL_DEBUG_BREAK_IF(true);
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