#include "nbl/asset/IRenderpass.h"

namespace nbl::asset
{
	
IRenderpass::IRenderpass(const SCreationParams& params, const CreationParamValidationResult& counts) : m_params(params),
    m_attachments(attachmentCount ? core::make_refctd_dynamic_array<attachments_array_t>(attachmentCount):nullptr),
    m_subpasses(subpassCount ? core::make_refctd_dynamic_array<subpasses_array_t>(subpassCount):nullptr),
    m_dependencies(dependencyCount ? core::make_refctd_dynamic_array<subpass_deps_array_t>(dependencyCount):nullptr)
{
    auto attachments = core::SRange<const SCreationParams::SAttachmentDescription>{params.attachments, params.attachments+params.attachmentCount};
    std::copy(attachments.begin(), attachments.end(), m_attachments->begin());
    m_params.attachments = m_attachments->data();

    auto subpasses = core::SRange<const SCreationParams::SSubpassDescription>{params.subpasses, params.subpasses+params.subpassCount};
    std::copy(subpasses.begin(), subpasses.end(), m_subpasses->begin());
    m_params.subpasses = m_subpasses->data();

    uint32_t attRefCnt = 0u;
    uint32_t preservedAttRefCnt = 0u;
    for (const auto& sb : (*m_subpasses))
    {
        attRefCnt += sb.colorAttachmentCount;
        attRefCnt += sb.inputAttachmentCount;
        if (sb.resolveAttachments)
            attRefCnt += sb.colorAttachmentCount;
        if (sb.depthStencilAttachment.attachment!=ATTACHMENT_UNUSED)
            ++attRefCnt;

        if (sb.preserveAttachments)
            preservedAttRefCnt += sb.preserveAttachmentCount;
    }
    if (attRefCnt)
        m_attachmentRefs = core::make_refctd_dynamic_array<attachment_refs_array_t>(attRefCnt);
    if (preservedAttRefCnt)
        m_preservedAttachmentRefs = core::make_refctd_dynamic_array<preserved_attachment_refs_array_t>(preservedAttRefCnt);

    uint32_t refOffset = 0u;
    uint32_t preservedRefOffset = 0u;
    auto* refs = m_attachmentRefs->data();
    auto* preservedRefs = m_preservedAttachmentRefs->data();
    for (auto& sb : (*m_subpasses))
    {
        if (m_attachmentRefs)
        {
#define _COPY_ATTACHMENT_REFS(_array,_count)\
            std::copy(sb._array, sb._array+sb._count, refs+refOffset);\
            sb._array = refs+refOffset;\
            refOffset += sb._count;

            _COPY_ATTACHMENT_REFS(colorAttachments, colorAttachmentCount);
            if (sb.inputAttachments)
            {
                _COPY_ATTACHMENT_REFS(inputAttachments, inputAttachmentCount);
            }
            if (sb.resolveAttachments)
            {
                _COPY_ATTACHMENT_REFS(resolveAttachments, colorAttachmentCount);
            }
            if (sb.depthStencilAttachment)
            {
                refs[refOffset] = sb.depthStencilAttachment[0];
                sb.depthStencilAttachment = refs + refOffset;
                ++refOffset;
            }
#undef _COPY_ATTACHMENT_REFS
        }

        if (m_preservedAttachmentRefs)
        {
            std::copy(sb.preserveAttachments, sb.preserveAttachments+sb.preserveAttachmentCount, preservedRefs+preservedRefOffset);
            sb.preserveAttachments = preservedRefs+preservedRefOffset;
            preservedRefOffset += sb.preserveAttachmentCount;
        }
    }

    if (!params.dependencies)
        return;

    auto deps = core::SRange<const SCreationParams::SSubpassDependency>{params.dependencies, params.dependencies+params.dependencyCount};
    std::copy(deps.begin(), deps.end(), m_dependencies->begin());
    m_params.dependencies = m_dependencies->data();
}

}