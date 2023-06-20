#ifndef _NBL_ASSET_I_FRAMBEBUFFER_H_INCLUDED_
#define _NBL_ASSET_I_FRAMBEBUFFER_H_INCLUDED_

#include "nbl/asset/IImageView.h"
#include "nbl/asset/IRenderpass.h"

namespace nbl::asset
{

template <typename RenderpassType, typename ImageViewType>
class IFramebuffer
{
    public:
        using renderpass_t = RenderpassType;
        using attachment_t = ImageViewType;

        struct SCreationParams
        {
            core::smart_refctd_ptr<renderpass_t> renderpass;
            core::smart_refctd_ptr<attachment_t>* depthStencilAttachments;
            core::smart_refctd_ptr<attachment_t>* colorAttachments;
            uint32_t width;
            uint32_t height;
            uint32_t layers;
        };

        static inline bool validate(const SCreationParams& params)
        {
            if (!params.width)
                return false;
            if (!params.height)
                return false;
            if (!params.layers)
                return false;

            const auto* const rp = params.renderpass.get();
            if (!rp)
                return false;
            const auto& rpParams = rp->getCreationParameters();

            /*
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a color
            attachment or resolve attachment by renderPass must have been created with a usage value including VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a depth/stencil attachment
            by renderPass must have been created with a usage value including VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as an input attachment
            by renderPass must have been created with a usage value including VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
            */
            auto invalidAttachments = [params](const uint32_t presentAttachments, const auto* const attachmentDesc, const core::smart_refctd_ptr<attachment_t>* const attachments, const IImage::E_USAGE_FLAGS usage) -> bool
            {
                for (uint32_t i=0u; i<presentAttachments; ++i)
                {
                    const auto& viewParams = attachments[i]->getCreationParameters();
                    if (viewParams.subresourceRange.layerCount<params.layers)
                        return true;
                    const auto& imgParams = viewParams.image->getCreationParameters();
                    if (imgParams.extent.width<params.width)
                        return false;
                    if (imgParams.extent.height<params.height)
                        return false;

                    const auto& desc = attachmentDesc[i];
                    if (viewParams.format!=desc.format || imgParams.samples!=desc.samples)
                        return true;
                    const auto realUsages = viewParams.actualUsages();
                    if (!viewParams.subUsages.hasFlags(usage))
                        return true;
                }
                return false;
            };
            if (invalidAttachments(rp->getDepthStencilAttachmentCount(),rpParams.depthStencilAttachments,params.depthStencilAttachments,IImage::EUF_DEPTH_STENCIL_ATTACHMENT_BIT))
                return false;
            if (invalidAttachments(rp->getColorAttachmentCount(),rpParams.colorAttachments,params.colorAttachments,IImage::EUF_COLOR_ATTACHMENT_BIT))
                return false;

            bool retval = true;
            core::visit_token_terminated_array(rp->getCreationParameters().subpasses,IRenderpass::SCreationParams::SubpassesEnd,[params,&retval](const IRenderpass::SCreationParams::SSubpassDescription& desc)->bool
            {
                core::visit_token_terminated_array(desc.inputAttachments,IRenderpass::SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&](const IRenderpass::SCreationParams::SSubpassDescription::SInputAttachmentRef& ia)->bool
                {
                        const auto& viewParams = (ia.isColor() ? params.colorAttachments[ia.asColor.attachmentIndex]:params.depthStencilAttachments[ia.asDepthStencil.attachmentIndex])->getCreationParameters();
                        if (viewParams.actualUsages().hasFlags(IImage::EUF_INPUT_ATTACHMENT_BIT))
                            return (retval=false);
                        return true;
                });
                return retval;
            });

            return retval;
        }

        const SCreationParams& getCreationParameters() const { return m_params; }

    protected:
        explicit IFramebuffer(SCreationParams&& params) : m_params(std::move(params))
        {
            const auto* const rp = params.renderpass.get();
            const auto depthStencilCount = rp->getDepthStencilAttachmentCount();
            const auto colorCount = rp->getColorAttachmentCount();
            m_attachments = core::make_refctd_dynamic_array<attachments_array_t>(depthStencilCount+colorCount);
            m_params.depthStencilAttachments = m_attachments->data();
            std::copy_n(params.depthStencilAttachments,depthStencilCount,m_params.depthStencilAttachments);
            m_params.colorAttachments = m_params.depthStencilAttachments+depthStencilCount;
            std::copy_n(params.colorAttachments,colorCount,m_params.colorAttachments);
        }
        virtual ~IFramebuffer() = default;

        SCreationParams m_params;
        using attachments_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<attachment_t>>;
        // storage for m_params.attachments
        attachments_array_t m_attachments;
};

}

#endif