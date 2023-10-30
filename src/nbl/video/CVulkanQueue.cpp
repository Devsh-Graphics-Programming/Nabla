#include "nbl/video/CVulkanQueue.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanSemaphore.h"
#include "nbl/video/CVulkanCommandBuffer.h"

namespace nbl::video
{

auto CVulkanQueue::waitIdle() const -> RESULT
{
    return getResultFrom(static_cast<const CVulkanLogicalDevice*>(m_originDevice)->getFunctionTable()->vk.vkQueueWaitIdle(m_vkQueue));
}
    
bool CVulkanQueue::startCapture() 
{
	if (!m_rdoc_api)
		return false;
    m_rdoc_api->StartFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(m_vkInstance), NULL);
	return true;
}

bool CVulkanQueue::endCapture()
{
	if (!m_rdoc_api)
		return false;
    m_rdoc_api->EndFrameCapture(RENDERDOC_DEVICEPOINTER_FROM_VKINSTANCE(m_vkInstance), NULL);
	return true;
}

<<<<<<< HEAD
auto CVulkanQueue::submit_impl(const uint32_t _count, const SSubmitInfo* const _submits) -> RESULT
{
    auto fillSemaphoreInfo = [this](const SSubmitInfo::SSemaphoreInfo* in, const uint32_t count, VkSemaphoreSubmitInfoKHR* out) -> void
    {
        for (uint32_t i=0u; i<count; i++)
        {
            out[i].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR;
            out[i].pNext = nullptr;
            out[i].semaphore = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(in[i],m_originDevice)->getInternalObject();
            out[i].value = in[i].value;
            out[i].stageMask = getVkPipelineStageFlagsFromPipelineStageFlags(in[i].stageMask);
            out[i].deviceIndex = 0u; // device groups are a TODO
        }
    };

    uint32_t waitSemCnt = 0u;
    uint32_t cmdBufCnt = 0u;
    uint32_t signalSemCnt = 0u;
    for (uint32_t i=0u; i<_count; ++i)
    {
        const auto& sb = _submits[i];
        waitSemCnt += sb.waitSemaphoreCount;
        cmdBufCnt += sb.commandBufferCount;
        signalSemCnt += sb.signalSemaphoreCount;
    }

    // TODO: we need a SVO optimized vector with SoA
    core::vector<VkSubmitInfo2> submits(_count,{VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,/*No interesting extensions*/nullptr,/*No protected stuff yet*/0});
    core::vector<VkSemaphoreSubmitInfoKHR> waitSemaphores(waitSemCnt);
    core::vector<VkCommandBufferSubmitInfoKHR> commandBuffers(cmdBufCnt,{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,nullptr});
    core::vector<VkSemaphoreSubmitInfoKHR> signalSemaphores(waitSemCnt);

    uint32_t waitSemOffset = 0u;
    uint32_t cmdBufOffset = 0u;
    uint32_t signalSemOffset = 0u;
    for (uint32_t i=0u; i<_count; ++i)
    {
        const SSubmitInfo& _sb = _submits[i];

        auto* waits = waitSemaphores.data()+waitSemOffset;
        waitSemOffset += _sb.waitSemaphoreCount;
        auto* cmdbufs = commandBuffers.data()+cmdBufOffset;
        cmdBufOffset += _sb.commandBufferCount;
        auto* signals = signalSemaphores.data()+signalSemOffset;
        signalSemOffset += _sb.signalSemaphoreCount;

        auto& sb = submits[i];
        sb.pWaitSemaphoreInfos = waits;
        sb.waitSemaphoreInfoCount = _sb.waitSemaphoreCount;
        sb.pCommandBufferInfos = cmdbufs;
        sb.commandBufferInfoCount = _sb.commandBufferCount;
        sb.pSignalSemaphoreInfos = signals;
        sb.signalSemaphoreInfoCount = _sb.signalSemaphoreCount;

        fillSemaphoreInfo(_sb.pWaitSemaphores,_sb.waitSemaphoreCount,waits);
        for (uint32_t j=0u; j<_sb.commandBufferCount; ++j)
        {
            cmdbufs[j].commandBuffer = IBackendObject::device_compatibility_cast<CVulkanCommandBuffer*>(_sb.commandBuffers[i],m_originDevice)->getInternalObject();
            cmdbufs[j].deviceMask = 0x1u;
        }
        fillSemaphoreInfo(_sb.pSignalSemaphores,_sb.signalSemaphoreCount,signals);
    }
    const auto vk_result = static_cast<const CVulkanLogicalDevice*>(m_originDevice)->getFunctionTable()->vk.vkQueueSubmit2KHR(m_vkQueue,_count,submits.data(),VK_NULL_HANDLE);
    return getResultFrom(vk_result);
=======
bool CVulkanQueue::insertDebugMarker(const char* name, const core::vector4df_SIMD& color)
{
    // This is instance function loaded by volk (via vkGetInstanceProcAddr), so we have to check for validity of the function ptr
    if (vkQueueInsertDebugUtilsLabelEXT == 0)
        return false;

    VkDebugUtilsLabelEXT labelInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
    labelInfo.pLabelName = name;
    labelInfo.color[0] = color.x;
    labelInfo.color[1] = color.y;
    labelInfo.color[2] = color.z;
    labelInfo.color[3] = color.w;

    vkQueueBeginDebugUtilsLabelEXT(m_vkQueue, &labelInfo);
    return true;
}

bool CVulkanQueue::beginDebugMarker(const char* name, const core::vector4df_SIMD& color)
{
    // This is instance function loaded by volk (via vkGetInstanceProcAddr), so we have to check for validity of the function ptr
    if (vkQueueBeginDebugUtilsLabelEXT == 0)
        return false;

    VkDebugUtilsLabelEXT labelInfo = { VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
    labelInfo.pLabelName = name;
    labelInfo.color[0] = color.x;
    labelInfo.color[1] = color.y;
    labelInfo.color[2] = color.z;
    labelInfo.color[3] = color.w;
    vkQueueBeginDebugUtilsLabelEXT(m_vkQueue, &labelInfo);

    return true;
}

bool CVulkanQueue::endDebugMarker()
{
    // This is instance function loaded by volk (via vkGetInstanceProcAddr), so we have to check for validity of the function ptr
    if (vkQueueEndDebugUtilsLabelEXT == 0)
        return false;
    vkQueueEndDebugUtilsLabelEXT(m_vkQueue);
    return true;
>>>>>>> 798939af864768c9d936d4810ae3718b8032f2c8
}

}