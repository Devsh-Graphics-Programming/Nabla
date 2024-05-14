#include "nbl/video/CVulkanDeferredOperation.h"
#include "nbl/video/CVulkanLogicalDevice.h"


namespace nbl::video
{

CVulkanDeferredOperation::~CVulkanDeferredOperation()
{
    assert(!isPending());
    const VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
    vkDestroyDeferredOperationKHR(vk_device,m_deferredOp,nullptr);
}
    
uint32_t CVulkanDeferredOperation::getMaxConcurrency() const
{
    const VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
    return vkGetDeferredOperationMaxConcurrencyKHR(vk_device,m_deferredOp);
}

auto CVulkanDeferredOperation::execute_impl() -> STATUS
{
    const VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
    switch (vkDeferredOperationJoinKHR(vk_device,m_deferredOp))
    {
        case VK_SUCCESS:
            return STATUS::COMPLETED;
            break;
        case VK_THREAD_DONE_KHR:
            return STATUS::THREAD_DONE;
            break;
        case VK_THREAD_IDLE_KHR:
            return STATUS::THREAD_IDLE;
            break;
        default:
            break;
    }
    return STATUS::_ERROR;
}
    
bool CVulkanDeferredOperation::isPending() const
{
    const VkDevice vk_device = static_cast<const CVulkanLogicalDevice*>(getOriginDevice())->getInternalObject();
    return vkGetDeferredOperationResultKHR(vk_device,m_deferredOp)==VK_NOT_READY;
}

void CVulkanDeferredOperation::operator delete(void* ptr) noexcept
{
    CVulkanDeferredOperation* cvkdo = reinterpret_cast<CVulkanDeferredOperation*>(ptr);
    auto vkDevice = static_cast<CVulkanLogicalDevice*>(const_cast<ILogicalDevice*>(cvkdo->getOriginDevice()));
    auto& mempool = vkDevice->getMemoryPoolForDeferredOperations();
    mempool.deallocate(ptr,sizeof(CVulkanDeferredOperation));
}

}