#include "nbl/video/CVulkanRenderpass.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

CVulkanRenderpass::CVulkanRenderpass(const SCreationParams& params) : IGPURenderpass(params)
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    VkRenderPassCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.pNext = nullptr;
    ci.attachmentCount = getAttachmentCount();
    ci.dependencyCount = getSubpassDependencies().size();
    ci.subpassCount = getSubpasses().size();
    ci.flags = static_cast<VkRenderPassCreateFlags>(0);
    core::vector<VkAttachmentDescription> attachments(ci.attachmentCount); // TODO reduce number of allocations/get rid of vectors
    uint32_t attachmentIndices[EAP_COUNT]{};
    for (uint32_t a = 0u, i = 0u; a < EAP_COUNT; ++a)
    {
        auto attachmentPoint = static_cast<E_ATTACHMENT_POINT>(a);
        if (!isAttachmentEnabled(attachmentPoint))
            continue;

        attachmentIndices[attachmentPoint] = i;

        const auto& att = m_params.attachments[attachmentPoint];
        auto& vkatt = attachments[i++];
        vkatt.finalLayout = static_cast<VkImageLayout>(att.finalLayout);
        vkatt.initialLayout = static_cast<VkImageLayout>(att.initialLayout);
        vkatt.flags = static_cast<VkAttachmentDescriptionFlags>(0);
        vkatt.format = static_cast<VkFormat>(att.format);
        vkatt.loadOp = static_cast<VkAttachmentLoadOp>(att.loadOp);
        vkatt.storeOp = static_cast<VkAttachmentStoreOp>(att.storeOp);
        vkatt.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        vkatt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        vkatt.samples = static_cast<VkSampleCountFlagBits>(att.samples);
    }
    ci.pAttachments = attachments.data();

    core::vector<VkSubpassDependency> deps(ci.dependencyCount);
    for (uint32_t i = 0u; i < deps.size(); ++i)
    {
        const auto& dep = m_params.dependencies[i];
        auto& vkdep = deps[i];

        vkdep.dependencyFlags = static_cast<VkDependencyFlags>(dep.dependencyFlags);
        vkdep.dstAccessMask = static_cast<VkAccessFlags>(dep.dstAccessMask);
        vkdep.dstStageMask = static_cast<VkPipelineStageFlags>(dep.dstStageMask);
        vkdep.srcAccessMask = static_cast<VkAccessFlags>(dep.srcAccessMask);
        vkdep.srcStageMask = static_cast<VkPipelineStageFlags>(dep.srcStageMask);
        vkdep.dstSubpass = dep.dstSubpass;
        vkdep.srcSubpass = dep.srcSubpass;
    }
    ci.pDependencies = deps.data();

    constexpr uint32_t AttachmentTypeCount = 4u; // w/o depthstencil
    constexpr uint32_t MaxDepthStencilAttachments = 2u;
    constexpr uint32_t mem_sz_per_subpass = ((EAP_COUNT*AttachmentTypeCount + MaxDepthStencilAttachments) * std::max(sizeof(uint32_t), sizeof(VkAttachmentReference)));

    const uint32_t mem_sz = ci.subpassCount * mem_sz_per_subpass;
    uint8_t* mem = reinterpret_cast<uint8_t*>( _NBL_ALIGNED_MALLOC(mem_sz, _NBL_SIMD_ALIGNMENT) );

    core::vector<VkSubpassDescription> subpasses(ci.subpassCount);
    for (uint32_t i = 0u; i < subpasses.size(); ++i)
    {
        const auto& sp = m_params.subpasses[i];
        auto& vksp = subpasses[i];

        VkAttachmentReference* subpass_mem = reinterpret_cast<VkAttachmentReference*>(mem + i*mem_sz_per_subpass);
        auto* a_depth = subpass_mem;
        uint32_t i_depth = 0u;
        auto* a_input = a_depth + MaxDepthStencilAttachments;
        uint32_t i_input = 0u;
        auto* a_color = a_input + EAP_COUNT;
        uint32_t i_color = 0u;
        auto* a_resolve = a_color + EAP_COUNT;
        uint32_t i_resolve = 0u;
        auto* a_preserved = reinterpret_cast<uint32_t*>(a_resolve + EAP_COUNT);
        uint32_t i_preserved = 0u;

        for (uint32_t j = 0u; j < EAP_COUNT; ++j)
        {
            auto point = static_cast<E_ATTACHMENT_POINT>(j);
            if (sp.references[point].usage == SCreationParams::SSubpassDescription::SAttachmentUsage::EU_UNUSED)
                continue;
            auto layout = static_cast<VkImageLayout>(sp.references[point].layout);

            VkAttachmentReference ref;
            ref.attachment = attachmentIndices[point];
            ref.layout = layout;

            // TODO:
            // erm ok this is fucked, 
            // this idea with attachments[EAP_COUNT] array, etc
            // there's no way to express preserved and resolve attachments then
            // they need separate arrays (user-provided ptrs most likely...)
            // which would end up in even more messy solution than vulkan api actually has
            switch (sp.references[point].usage)
            {
            case SCreationParams::SSubpassDescription::SAttachmentUsage::EU_PRESERVED:
                a_preserved[i_preserved++] = ref.attachment;
                continue;
                break;
            case SCreationParams::SSubpassDescription::SAttachmentUsage::EU_RESOLVE:
                a_resolve[i_resolve++] = ref;
                continue;
                break;
            }

            if (isDepthOrStencilAttachmentPoint(point))
            {
                a_depth[i_depth++] = ref;
            }
            else if (isColorAttachmentPoint(point))
            {
                a_color[i_color++] = ref;
            }
            else if (isInputAttachmentPoint(point))
            {
                a_input[i_input++] = ref;
            }
        }

        /*
        If flags does not include VK_SUBPASS_DESCRIPTION_SHADER_RESOLVE_BIT_QCOM, 
        and if pResolveAttachments is not NULL, each of its elements corresponds to a color attachment (the element in pColorAttachments at the same index)*/
        // ^^^^^^^ this is a problem with resolve atatchments!! (TODO)
        vksp.flags = static_cast<VkSubpassDescriptionFlags>(sp.flags);
        vksp.colorAttachmentCount = i_color;
        vksp.inputAttachmentCount = i_input;
        vksp.preserveAttachmentCount = i_preserved;
        vksp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        vksp.pColorAttachments = a_color;
        vksp.pDepthStencilAttachment = a_depth;
        vksp.pInputAttachments = a_input;
        vksp.pResolveAttachments = a_resolve;
        vksp.pPreserveAttachments = a_preserved;
    }
    ci.pSubpasses = subpasses.data();

    vk->vk.vkCreateRenderPass(vkdev, &ci, nullptr, &m_renderpass);
}

CVulkanRenderpass::~CVulkanRenderpass()
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    vk->vk.vkDestroyRenderPass(vkdev, m_renderpass, nullptr);
}

}
}