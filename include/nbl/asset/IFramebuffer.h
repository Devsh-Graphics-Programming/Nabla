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
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-width-00885
            if (!params.width)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-width-00887
            if (!params.height)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-width-00889
            if (!params.layers)
                return false;

            const auto* const rp = params.renderpass.get();
            if (!rp)
                return false;
            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-renderPass-02531
            if (rp->hasViewMasks() && params.layers!=1)
                return false;

            // provoke wraparound of -1 on purpose
            const uint32_t viewMaskMSB = static_cast<uint32_t>(rp->getViewMaskMSB());
            /*
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a color
            attachment or resolve attachment by renderPass must have been created with a usage value including VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a depth/stencil attachment
            by renderPass must have been created with a usage value including VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as an input attachment
            by renderPass must have been created with a usage value including VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
            */
            auto invalidAttachments = [rp,params,viewMaskMSB](const uint32_t presentAttachments, const auto* const attachmentDesc, const core::smart_refctd_ptr<attachment_t>* const attachments) -> bool
            {
                for (uint32_t i=0u; i<presentAttachments; ++i)
                {
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-02778
                    if (!attachments[i])
                        return true;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-commonparent
                    if (rp->isCompatibleDevicewise(attachments[i].get()))
                        return true;
                    
                    const auto& viewParams = attachments[i]->getCreationParameters();

                    const auto& subresourceRange = viewParams.subresourceRange;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-04535
                    if (subresourceRange.layerCount<params.layers)
                        return true;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-renderPass-04536
                    if (subresourceRange.layerCount<=viewMaskMSB)
                        return true;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00883
                    if (subresourceRange.levelCount!=1)
                        return true;

                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00884
                    if (viewParams.components!=attachment_t::SComponentMapping())
                        return true;

                    const auto& imgParams = viewParams.image->getCreationParameters();
                    // We won't support linear attachments
                    if (imgParams->getTiling()!=IGPUImage::TILING::OPTIMAL)
                        return true;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-04533
                    if (imgParams.extent.width<params.width)
                        return true;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-04534
                    if (imgParams.extent.height<params.height)
                        return true;

                    const auto& desc = attachmentDesc[i];
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00880
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00881
                    if (viewParams.format!=desc.format || imgParams.samples!=desc.samples)
                        return true;

                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00877
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-02633
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-02634
                    if (!viewParams.actualUsages().hasFlags(IImage::EUF_RENDER_ATTACHMENT_BIT))
                        return true;

                    const auto viewType = viewParams.viewType;
                    // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00891
                    if constexpr (std::is_same_v<std::remove_pointer_t<decltype(attachmentDesc)>,const IRenderpass::SCreationParams::SDepthStencilAttachmentDescription>)
                    if (imgParams.type==IImage::ET_3D && (viewType==attachment_t::ET_2D||viewType==attachment_t::ET_2D_ARRAY))
                        return true;
                }
                return false;
            };
            const auto& rpParams = rp->getCreationParameters();
            if (invalidAttachments(rp->getDepthStencilAttachmentCount(),rpParams.depthStencilAttachments,params.depthStencilAttachments))
                return false;
            if (invalidAttachments(rp->getColorAttachmentCount(),rpParams.colorAttachments,params.colorAttachments))
                return false;

            bool retval = true;
            core::visit_token_terminated_array(rp->getCreationParameters().subpasses,IRenderpass::SCreationParams::SubpassesEnd,[params,&retval](const IRenderpass::SCreationParams::SSubpassDescription& desc)->bool
            {
                core::visit_token_terminated_array(desc.inputAttachments,IRenderpass::SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&](const IRenderpass::SCreationParams::SSubpassDescription::SInputAttachmentRef& ia)->bool
                {
                        const auto& viewParams = (ia.isColor() ? params.colorAttachments[ia.asColor.attachmentIndex]:params.depthStencilAttachments[ia.asDepthStencil.attachmentIndex])->getCreationParameters();
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00879
                        if (viewParams.actualUsages().hasFlags(IImage::EUF_INPUT_ATTACHMENT_BIT))
                            return (retval=false);
                        //TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-samples-07009
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