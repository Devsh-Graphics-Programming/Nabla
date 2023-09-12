#include "nbl/asset/IRenderpass.h"

namespace nbl::asset
{
	
IRenderpass::IRenderpass(const SCreationParams& params, const SCreationParamValidationResult& counts) : m_params(params),
    m_depthStencilAttachments(counts.depthStencilAttachmentCount ? core::make_refctd_dynamic_array<depth_stencil_attachments_array_t>(counts.depthStencilAttachmentCount):nullptr),
    m_colorAttachments(counts.colorAttachmentCount ? core::make_refctd_dynamic_array<color_attachments_array_t>(counts.colorAttachmentCount):nullptr),
    m_subpasses(core::make_refctd_dynamic_array<subpass_array_t>(counts.subpassCount)),
    m_inputAttachments(core::make_refctd_dynamic_array<subpass_array_t>(counts.totalInputAttachmentCount+counts.subpassCount)),
    m_preserveAttachments(core::make_refctd_dynamic_array<subpass_array_t>(counts.totalPreserveAttachmentCount+counts.subpassCount)),
    m_subpassDependencies(counts.dependencyCount ? core::make_refctd_dynamic_array<subpass_deps_array_t>(counts.dependencyCount):nullptr)
{
    m_params.depthStencilAttachments = m_depthStencilAttachments ? m_depthStencilAttachments->data():(&SCreationParams::DepthStencilAttachmentsEnd);
    for (auto i=0u; i<counts.depthStencilAttachmentCount; i++)
    {
        const auto& attachment = params.depthStencilAttachments[i];
        if (attachment.loadOp.depth==LOAD_OP::LOAD || attachment.loadOp.actualStencilOp()==LOAD_OP::LOAD)
            m_loadOpDepthStencilAttachmentEnd = i;
        m_depthStencilAttachments->operator[](i) = attachment;
    }
    m_loadOpDepthStencilAttachmentEnd++;

    m_params.colorAttachments = m_colorAttachments ? m_colorAttachments->data():(&SCreationParams::ColorAttachmentsEnd);
    for (auto i=0u; i<counts.colorAttachmentCount; i++)
    {
        const auto& attachment = params.colorAttachments[i];
        if (attachment.loadOp==LOAD_OP::LOAD)
            m_loadOpColorAttachmentEnd = i;
        m_colorAttachments->operator[](i) = attachment;
    }
    m_loadOpColorAttachmentEnd++;

    m_params.subpasses = m_subpasses->data();
    {
        auto oit = m_subpasses->data();
        auto inputAttachments = m_inputAttachments->data();
        auto preserveAttachments = m_preserveAttachments->data();
        core::visit_token_terminated_array(params.subpasses,SCreationParams::SubpassesEnd,[&oit,&inputAttachments,&preserveAttachments](const SCreationParams::SSubpassDescription& desc)->bool
        {
            *oit = desc;
            oit->inputAttachments = inputAttachments;
            core::visit_token_terminated_array(desc.inputAttachments,SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&inputAttachments](const auto& ia)->bool
            {
                *(inputAttachments++) = ia;
            });
            *(inputAttachments++) = SCreationParams::SSubpassDescription::InputAttachmentsEnd;
            oit->preserveAttachments = preserveAttachments;
            core::visit_token_terminated_array(desc.preserveAttachments,SCreationParams::SSubpassDescription::AttachmentUnused,[&preserveAttachments](const auto& pa)->bool
            {
                *(preserveAttachments++) = pa;
            });
            *(preserveAttachments++) = SCreationParams::SSubpassDescription::AttachmentUnused;
            oit++;
            return true;
        });
    };

    m_params.dependencies = m_subpassDependencies ? m_subpassDependencies->data():(&SCreationParams::DependenciesEnd);
    std::copy_n(params.dependencies,counts.dependencyCount,m_params.dependencies);
}

}