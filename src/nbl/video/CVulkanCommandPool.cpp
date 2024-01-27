#include "nbl/video/CVulkanCommandPool.h"
#include "nbl/video/CVulkanCommandBuffer.h"
#include "nbl/video/CVulkanLogicalDevice.h"

namespace nbl::video
{

CVulkanCommandPool::~CVulkanCommandPool()
{
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    vulkanDevice->getFunctionTable()->vk.vkDestroyCommandPool(vulkanDevice->getInternalObject(),m_vkCommandPool,nullptr);
}

void CVulkanCommandPool::trim()
{
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    vulkanDevice->getFunctionTable()->vk.vkTrimCommandPool(vulkanDevice->getInternalObject(),m_vkCommandPool,0u);
}

bool CVulkanCommandPool::createCommandBuffers_impl(const BUFFER_LEVEL level, const std::span<core::smart_refctd_ptr<IGPUCommandBuffer>> outCmdBufs, core::smart_refctd_ptr<system::ILogger>&& logger)
{
    const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());

	StackAllocation<VkCommandBuffer> vk_commandBuffers(this,outCmdBufs.size());
    if (!vk_commandBuffers)
        return false;

    VkCommandBufferAllocateInfo info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,nullptr };
    info.commandPool = m_vkCommandPool;
    info.level = static_cast<VkCommandBufferLevel>(level);
    info.commandBufferCount = vk_commandBuffers.size();
	if (vulkanDevice->getFunctionTable()->vk.vkAllocateCommandBuffers(vulkanDevice->getInternalObject(),&info,vk_commandBuffers.data())!=VK_SUCCESS)
		return false;
	
    if (!logger)
    {
        auto debugCB = getOriginDevice()->getPhysicalDevice()->getDebugCallback();
        if (debugCB)
            logger = core::smart_refctd_ptr<system::ILogger>(debugCB->getLogger());
    }
    for (auto i=0u; i<outCmdBufs.size(); ++i)
        outCmdBufs[i] = core::make_smart_refctd_ptr<CVulkanCommandBuffer>(core::smart_refctd_ptr<const ILogicalDevice>(vulkanDevice),level,vk_commandBuffers[i],core::smart_refctd_ptr<IGPUCommandPool>(this),std::move(logger));
	return true;
}

void CVulkanCommandPool::setObjectDebugName(const char* label) const
{
    IBackendObject::setObjectDebugName(label);

	if (!vkSetDebugUtilsObjectNameEXT)
		return;

    const CVulkanLogicalDevice* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
	VkDebugUtilsObjectNameInfoEXT nameInfo = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, nullptr};
	nameInfo.objectType = VK_OBJECT_TYPE_COMMAND_POOL;
	nameInfo.objectHandle = reinterpret_cast<uint64_t>(getInternalObject());
	nameInfo.pObjectName = getObjectDebugName();
	vkSetDebugUtilsObjectNameEXT(vulkanDevice->getInternalObject(), &nameInfo);
}

bool CVulkanCommandPool::reset_impl()
{
	const auto* vulkanDevice = static_cast<const CVulkanLogicalDevice*>(getOriginDevice());
    return vulkanDevice->getFunctionTable()->vk.vkResetCommandPool(vulkanDevice->getInternalObject(),m_vkCommandPool,VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT)==VK_SUCCESS;
}

}