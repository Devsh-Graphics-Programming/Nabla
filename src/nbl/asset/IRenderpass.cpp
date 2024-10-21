#include "nbl/asset/IRenderpass.h"

namespace nbl::asset
{
	
IRenderpass::IRenderpass(const SCreationParams& params, const SCreationParamValidationResult& counts) : m_params(params),
    m_depthStencilAttachments(counts.depthStencilAttachmentCount ? core::make_refctd_dynamic_array<depth_stencil_attachments_array_t>(counts.depthStencilAttachmentCount+1):nullptr),
    m_colorAttachments(counts.colorAttachmentCount ? core::make_refctd_dynamic_array<color_attachments_array_t>(counts.colorAttachmentCount+1):nullptr),
    m_subpasses(core::make_refctd_dynamic_array<subpass_array_t>(counts.subpassCount+1)),
    m_inputAttachments(core::make_refctd_dynamic_array<input_attachment_array_t>(counts.totalInputAttachmentCount+counts.subpassCount)),
    m_preserveAttachments(core::make_refctd_dynamic_array<preserved_attachment_refs_array_t>(counts.totalPreserveAttachmentCount+counts.subpassCount)),
    m_subpassDependencies(counts.dependencyCount ? core::make_refctd_dynamic_array<subpass_deps_array_t>(counts.dependencyCount+1):nullptr),
    m_viewMaskMSB(counts.viewMaskMSB)
{
    m_params.depthStencilAttachments = m_depthStencilAttachments ? m_depthStencilAttachments->data():(&SCreationParams::DepthStencilAttachmentsEnd);
    for (auto i=0u; i<counts.depthStencilAttachmentCount; i++)
    {
        const auto& attachment = params.depthStencilAttachments[i];
        if (attachment.loadOp.depth==LOAD_OP::LOAD || attachment.loadOp.actualStencilOp()==LOAD_OP::LOAD)
            m_loadOpDepthStencilAttachmentEnd = i;
        m_depthStencilAttachments->operator[](i) = attachment;
    }
    if (m_depthStencilAttachments)
        m_depthStencilAttachments->back() = SCreationParams::DepthStencilAttachmentsEnd;
    m_loadOpDepthStencilAttachmentEnd++;

    m_params.colorAttachments = m_colorAttachments ? m_colorAttachments->data():(&SCreationParams::ColorAttachmentsEnd);
    for (auto i=0u; i<counts.colorAttachmentCount; i++)
    {
        const auto& attachment = params.colorAttachments[i];
        if (attachment.loadOp==LOAD_OP::LOAD)
            m_loadOpColorAttachmentEnd = i;
        m_colorAttachments->operator[](i) = attachment;
    }
    if (m_colorAttachments)
        m_colorAttachments->back() = SCreationParams::ColorAttachmentsEnd;
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
                return true;
            });
            *(inputAttachments++) = SCreationParams::SSubpassDescription::InputAttachmentsEnd;
            oit->preserveAttachments = preserveAttachments;
            core::visit_token_terminated_array(desc.preserveAttachments,SCreationParams::SSubpassDescription::PreserveAttachmentsEnd,[&preserveAttachments](const auto& pa)->bool
            {
                *(preserveAttachments++) = pa;
                return true;
            });
            *(preserveAttachments++) = SCreationParams::SSubpassDescription::PreserveAttachmentsEnd;
            oit++;
            return true;
        });
        *oit = SCreationParams::SubpassesEnd;
    };

    if (m_subpassDependencies)
    {
        std::copy_n(params.dependencies,counts.dependencyCount,m_subpassDependencies->data());
        m_subpassDependencies->back() = SCreationParams::DependenciesEnd;
        m_params.dependencies = m_subpassDependencies->data();
    }
    else
        m_params.dependencies = &SCreationParams::DependenciesEnd;
}

}