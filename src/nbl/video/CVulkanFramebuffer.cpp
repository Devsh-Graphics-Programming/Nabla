#include "nbl/video/CVulkanFramebuffer.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

CVulkanFramebuffer::CVulkanFramebuffer(CVKLogicalDevice* _vkdev, SCreationParams&& params) : IFramebuffer(std::move(params)), m_vkdevice(_vkdev)
{
    VkFramebufferCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.pNext = nullptr;
    ci.attachmentCount = m_params.getPresentAttachmentCount();
    ci.flags = static_cast<VkFramebufferCreateFlags>(m_params.flags);
    ci.width = m_params.width;
    ci.height = m_params.height;
    ci.layers = m_params.layers;
    //ci.pAttachments // TODO views vk handles
    //ci.renderPass //TODO

    auto vkdev = _vkdev->getInternalObject();
    auto* vk = _vkdev->getFunctionTable();

    vk->vk.vkCreateFramebuffer(vkdev, &ci, nullptr, &m_vkfbo);
}

CVulkanFramebuffer::~CVulkanFramebuffer()
{
    auto vkdev = m_vkdevice->getInternalObject();
    auto* vk = m_vkdevice->getFunctionTable();

    vk->vk.vkDestroyFramebuffer(vkdev, m_vkfbo, nullptr);
}

}
}