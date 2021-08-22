#include "CVulkanDeferredOperation.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl::video
{
CVulkanDeferredOperation::~CVulkanDeferredOperation()
{
    assert(E_STATUS::ES_COMPLETED == getStatus());
    const auto originDevice = getOriginDevice();
    VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
    vkDestroyDeferredOperationKHR(vk_device, m_deferredOp, nullptr);
}

bool CVulkanDeferredOperation::join() {
    const auto originDevice = getOriginDevice();
    VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
    VkResult vk_res = vkDeferredOperationJoinKHR(vk_device, m_deferredOp);
    return (VK_SUCCESS == vk_res);
}
    
uint32_t CVulkanDeferredOperation::getMaxConcurrency() {
    const auto originDevice = getOriginDevice();
    VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
    uint32_t res = vkGetDeferredOperationMaxConcurrencyKHR(vk_device, m_deferredOp);
    return res;
}
    
IDeferredOperation::E_STATUS CVulkanDeferredOperation::getStatus() {
    const auto originDevice = getOriginDevice();
    VkDevice vk_device = static_cast<const CVKLogicalDevice*>(originDevice)->getInternalObject();
    VkResult vk_res = vkGetDeferredOperationResultKHR(vk_device, m_deferredOp);
    if(VK_SUCCESS == vk_res) {
        return E_STATUS::ES_COMPLETED;
    } else if (VK_NOT_READY == vk_res) {
        return E_STATUS::ES_NOT_READY;
    }  else if (VK_THREAD_DONE_KHR == vk_res) {
        return E_STATUS::ES_THREAD_DONE;
    }  else if (VK_THREAD_IDLE_KHR == vk_res) {
        return E_STATUS::ES_THREAD_IDLE;
    } else {
        assert(false && "This case is not handled.");
        return E_STATUS::ES_NOT_READY;
    }
}

}