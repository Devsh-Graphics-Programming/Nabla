#include "nbl/video/CVulkanFramebuffer.h"

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"

namespace nbl::video
{

CVulkanFramebuffer::CVulkanFramebuffer(CVKLogicalDevice* _vkdev, SCreationParams&& params) : IGPUFramebuffer(_vkdev, std::move(params)), m_vkdevice(_vkdev)
{
    // Todo(achal): Handle it properly
    assert(!(m_params.flags & ECF_IMAGELESS_BIT));

    constexpr uint32_t MemSize = 1u<<12;
    constexpr uint32_t MaxAttachments = MemSize / sizeof(VkImageView);

    VkImageView attachments[MaxAttachments];
    for (uint32_t i = 0u; i < m_params.attachmentCount; ++i)
    {
        auto* a = static_cast<CVulkanImageView*>(m_params.attachments[i].get());
        attachments[i] = a->getInternalObject();
    }

    VkFramebufferCreateInfo createInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    createInfo.pNext = nullptr;
    createInfo.flags = static_cast<VkFramebufferCreateFlags>(m_params.flags);
    createInfo.renderPass = static_cast<CVulkanRenderpass*>(m_params.renderpass.get())->getInternalObject();
    createInfo.attachmentCount = m_params.attachmentCount;
    assert(createInfo.attachmentCount <= MaxAttachments);
    createInfo.pAttachments = attachments;
    createInfo.width = m_params.width;
    createInfo.height = m_params.height;
    createInfo.layers = m_params.layers;

    auto vkdev = _vkdev->getInternalObject();
    // auto* vk = _vkdev->getFunctionTable();

    // vk->vk.vkCreateFramebuffer(vkdev, &createInfo, nullptr, &m_vkfbo);
    vkCreateFramebuffer(vkdev, &createInfo, nullptr, &m_vkfbo);
}

CVulkanFramebuffer::~CVulkanFramebuffer()
{
    auto vkdev = m_vkdevice->getInternalObject();
    // auto* vk = m_vkdevice->getFunctionTable();

    // vk->vk.vkDestroyFramebuffer(vkdev, m_vkfbo, nullptr);
    vkDestroyFramebuffer(vkdev, m_vkfbo, nullptr);
}

}