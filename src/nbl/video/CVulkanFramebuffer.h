#ifndef __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__
#define __NBL_C_VULKAN_FRAMEBUFFER_H_INCLUDED__

#include "nbl/video/IGPUFramebuffer.h"

#include <volk.h>

namespace nbl::video
{

class ILogicalDevice;

class CVulkanFramebuffer final : public IGPUFramebuffer
{
public:
    CVulkanFramebuffer(core::smart_refctd_ptr<ILogicalDevice>&& dev, SCreationParams&& params)
        : IGPUFramebuffer(std::move(dev), std::move(params))
    {
#if 0
        // Todo(achal): Handle it properly
        assert(!(m_params.flags & ECF_IMAGELESS_BIT));

        constexpr uint32_t MemSize = 1u << 12;
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
#endif
    }

    ~CVulkanFramebuffer();

private:
    VkFramebuffer m_vkfbo;
};

}

#endif