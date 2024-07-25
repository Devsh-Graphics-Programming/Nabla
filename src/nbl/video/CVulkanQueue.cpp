#include "nbl/video/CVulkanQueue.h"

#include "nbl/video/CVulkanLogicalDevice.h"
#include "nbl/video/CVulkanSemaphore.h"
#include "nbl/video/CVulkanCommandBuffer.h"

namespace nbl::video
{

auto CVulkanQueue::waitIdle_impl() const -> RESULT
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

auto CVulkanQueue::submit_impl(const std::span<const IQueue::SSubmitInfo> _submits) -> RESULT
{
    auto fillSemaphoreInfo = [this](const std::span<const SSubmitInfo::SSemaphoreInfo> semaphores, VkSemaphoreSubmitInfoKHR* &out) -> uint32_t
    {
        auto old = out;
        for (const auto& in : semaphores)
        {
            out->sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR;
            out->pNext = nullptr;
            out->semaphore = IBackendObject::device_compatibility_cast<CVulkanSemaphore*>(in.semaphore,m_originDevice)->getInternalObject();
            out->value = in.value;
            out->stageMask = getVkPipelineStageFlagsFromPipelineStageFlags(in.stageMask);
            out->deviceIndex = 0u; // device groups are a TODO
            out++;
        }
        return static_cast<uint32_t>(std::distance(old,out));
    };

    uint32_t waitSemCnt = 0u;
    uint32_t cmdBufCnt = 0u;
    uint32_t signalSemCnt = 0u;
    for (const auto& submit : _submits)
    {
        waitSemCnt += submit.waitSemaphores.size();
        cmdBufCnt += submit.commandBuffers.size();
        signalSemCnt += submit.signalSemaphores.size();
    }

    // TODO: we need a SVO optimized vector with SoA
    core::vector<VkSubmitInfo2> submits(_submits.size(),{VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR,/*No interesting extensions*/nullptr,/*No protected stuff yet*/0});
    core::vector<VkSemaphoreSubmitInfoKHR> waitSemaphores(waitSemCnt);
    core::vector<VkCommandBufferSubmitInfoKHR> commandBuffers(cmdBufCnt,{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR,nullptr});
    core::vector<VkSemaphoreSubmitInfoKHR> signalSemaphores(signalSemCnt);

    auto outSubmitInfo = submits.data();
    auto outWaitSemaphoreInfo = waitSemaphores.data();
    auto outCommandBufferInfo = commandBuffers.data();
    auto outSignalSemaphoreInfo = signalSemaphores.data();
    for (const auto& submit : _submits)
    {
        outSubmitInfo->pWaitSemaphoreInfos = outWaitSemaphoreInfo;
        outSubmitInfo->pCommandBufferInfos = outCommandBufferInfo;
        outSubmitInfo->pSignalSemaphoreInfos = outSignalSemaphoreInfo;

        for (const auto& commandBuffer : submit.commandBuffers)
        {
            outCommandBufferInfo->commandBuffer = IBackendObject::device_compatibility_cast<CVulkanCommandBuffer*>(commandBuffer.cmdbuf,m_originDevice)->getInternalObject();
            outCommandBufferInfo->deviceMask = 0x1u;
            outCommandBufferInfo++;
        }

        outSubmitInfo->waitSemaphoreInfoCount = fillSemaphoreInfo(submit.waitSemaphores,outWaitSemaphoreInfo);
        outSubmitInfo->commandBufferInfoCount = submit.commandBuffers.size();
        outSubmitInfo->signalSemaphoreInfoCount = fillSemaphoreInfo(submit.signalSemaphores,outSignalSemaphoreInfo);
        outSubmitInfo++;
    }
    const auto vk_result = static_cast<const CVulkanLogicalDevice*>(m_originDevice)->getFunctionTable()->vk.vkQueueSubmit2(m_vkQueue,submits.size(),submits.data(),VK_NULL_HANDLE);
    return getResultFrom(vk_result);
}

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
}


}