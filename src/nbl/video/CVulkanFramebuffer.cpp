#include "nbl/video/CVulkanFramebuffer.h"

#include "nbl/video/CVKLogicalDevice.h"
#include "nbl/video/CVulkanRenderpass.h"
#include "nbl/video/CVulkanImageView.h"

namespace nbl {
namespace video
{

CVulkanFramebuffer::CVulkanFramebuffer(CVKLogicalDevice* _vkdev, SCreationParams&& params) : IGPUFramebuffer(_vkdev, std::move(params)), m_vkdevice(_vkdev)
{
    constexpr uint32_t MemSize = 1u<<12;
    constexpr uint32_t MaxAttachments = MemSize / sizeof(VkImageView);

    VkImageView attachments[MaxAttachments];
    if (!(m_params.flags & ECF_IMAGELESS_BIT))
    {
        for (uint32_t i = 0u; i < m_params.attachmentCount; ++i)
        {
            auto* a = static_cast<CVulkanImageView*>(m_params.attachments[i].get());
            attachments[i] = a->getInternalObject();
        }
    }

    VkFramebufferCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    ci.pNext = nullptr;
    ci.attachmentCount = m_params.attachmentCount;
    assert(ci.attachmentCount <= MaxAttachments);
    ci.pAttachments = attachments;
    ci.flags = static_cast<VkFramebufferCreateFlags>(m_params.flags);
    ci.width = m_params.width;
    ci.height = m_params.height;
    ci.layers = m_params.layers;
    ci.renderPass = static_cast<CVulkanRenderpass*>(m_params.renderpass.get())->getInternalObject();

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