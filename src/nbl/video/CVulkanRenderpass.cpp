#include "nbl/video/CVulkanRenderpass.h"

#include "nbl/video/CVKLogicalDevice.h"

namespace nbl {
namespace video
{

CVulkanRenderpass::CVulkanRenderpass(CVKLogicalDevice* logicalDevice, const SCreationParams& params)
    : IGPURenderpass(logicalDevice, params), m_vkdev(logicalDevice)
{
    auto* vk = m_vkdev->getFunctionTable();
    auto vkdev = m_vkdev->getInternalObject();

    VkRenderPassCreateInfo ci;
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.pNext = nullptr;
    ci.attachmentCount = getAttachments().size();
    ci.dependencyCount = getSubpassDependencies().size();
    ci.subpassCount = getSubpasses().size();
    ci.flags = static_cast<VkRenderPassCreateFlags>(0);
    core::vector<VkAttachmentDescription> attachments(ci.attachmentCount); // TODO reduce number of allocations/get rid of vectors
    for (uint32_t i = 0u; i < attachments.size(); ++i)
    {
        const auto& att = m_params.attachments[i];
        auto& vkatt = attachments[i];
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

    core::vector<VkSubpassDescription> subpasses(ci.subpassCount);
    constexpr uint32_t MemSz = 1u<<12;
    constexpr uint32_t MaxAttachmentRefs = MemSz / sizeof(VkAttachmentReference);
    VkAttachmentReference attRefs[MaxAttachmentRefs];
    for (uint32_t i = 0u; i < m_attachmentRefs->size(); ++i)
    {
        auto& vkref = attRefs[i];
        const auto& ref = (*m_attachmentRefs)[i];

        vkref.attachment = ref.attachment;
        vkref.layout = static_cast<VkImageLayout>(ref.layout);
    }

    for (uint32_t i = 0u; i < subpasses.size(); ++i)
    {
        const auto& sp = m_params.subpasses[i];
        auto& vksp = subpasses[i];

        auto myRefsBegin = m_attachmentRefs->cbegin();

        vksp.flags = static_cast<VkSubpassDescriptionFlags>(sp.flags);
        vksp.colorAttachmentCount = sp.colorAttachmentCount;
        vksp.inputAttachmentCount = sp.inputAttachmentCount;
        vksp.preserveAttachmentCount = sp.preserveAttachmentCount;
        vksp.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        vksp.pColorAttachments = sp.colorAttachments ? (attRefs + (sp.colorAttachments - myRefsBegin)) : nullptr;
        vksp.pDepthStencilAttachment = sp.colorAttachments ? (attRefs + (sp.depthStencilAttachment - myRefsBegin)) : nullptr;
        vksp.pInputAttachments = sp.inputAttachments ? (attRefs + (sp.inputAttachments - myRefsBegin)) : nullptr;
        vksp.pResolveAttachments = sp.resolveAttachments ? (attRefs + (sp.resolveAttachments - myRefsBegin)) : nullptr;
        vksp.pPreserveAttachments = sp.preserveAttachments;
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