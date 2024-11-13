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

        template<bool Creation>
        struct SCreationParamsBase
        {
            using attachment_ptr_t = std::conditional_t<Creation,ImageViewType*const *,core::smart_refctd_ptr<ImageViewType>*>;

            core::smart_refctd_ptr<renderpass_t> renderpass;
            attachment_ptr_t depthStencilAttachments = nullptr;
            attachment_ptr_t colorAttachments = nullptr;
            uint32_t width;
            uint32_t height;
            uint32_t layers = 1;
        };
        struct SCreationParams : SCreationParamsBase<true>
        {
            using base_t = SCreationParamsBase<true>;

            inline bool validate() const
            {
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-width-00885
                if (!base_t::width)
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-width-00887
                if (!base_t::height)
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-width-00889
                if (!base_t::layers)
                    return false;

                const auto* const rp = base_t::renderpass.get();
                if (!rp)
                    return false;
                // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-renderPass-02531
                if (rp->hasViewMasks() && base_t::layers!=1)
                    return false;

                auto invalidUsageForLayout = [](const IImage::LAYOUT layout, const core::bitflag<IImage::E_USAGE_FLAGS> usages) -> bool
                {
                    switch (layout)
                    {
                        case IImage::LAYOUT::READ_ONLY_OPTIMAL:
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-03097
                            if (!(usages&(IImage::E_USAGE_FLAGS::EUF_SAMPLED_BIT|IImage::E_USAGE_FLAGS::EUF_INPUT_ATTACHMENT_BIT)))
                                return true;
                            break;
                        case IImage::LAYOUT::ATTACHMENT_OPTIMAL:
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-03094
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-03096
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-02844
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-stencilInitialLayout-02845
                            if (!usages.hasFlags(IImage::E_USAGE_FLAGS::EUF_RENDER_ATTACHMENT_BIT))
                                return true;
                            break;
                        case IImage::LAYOUT::TRANSFER_SRC_OPTIMAL:
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-03098
                            if (!usages.hasFlags(IImage::E_USAGE_FLAGS::EUF_TRANSFER_SRC_BIT))
                                return true;
                            break;
                        case IImage::LAYOUT::TRANSFER_DST_OPTIMAL:
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-03099
                            if (!usages.hasFlags(IImage::E_USAGE_FLAGS::EUF_TRANSFER_DST_BIT))
                                return true;
                            break;
                        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-07002
                        // TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBeginRenderPass2KHR.html#VUID-vkCmdBeginRenderPass2-initialLayout-07003
                        default:
                            break;
                    }
                    return false;
                };

                // upgrade to 32bit on purpose
                const int32_t viewMaskMSB = rp->getViewMaskMSB();
                /*
                * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a color
                attachment or resolve attachment by renderPass must have been created with a usage value including VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a depth/stencil attachment
                by renderPass must have been created with a usage value including VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as an input attachment
                by renderPass must have been created with a usage value including VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
                */
                auto invalidAttachments = [&](const uint32_t presentAttachments, const auto* const attachmentDesc, const ImageViewType* const* const attachments) -> bool
                {
                    for (uint32_t i=0u; i<presentAttachments; ++i)
                    {
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-02778
                        if (!attachments[i])
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-commonparent
                        if (!rp->isCompatibleDevicewise(attachments[i]))
                            return true;
                    
                        const auto& viewParams = attachments[i]->getCreationParameters();
                        const auto& imgParams = viewParams.image->getCreationParameters();

                        const auto& subresourceRange = viewParams.subresourceRange;
                        const int32_t layerCount = static_cast<int32_t>(subresourceRange.layerCount!=ImageViewType::remaining_array_layers ? subresourceRange.layerCount:imgParams.arrayLayers);
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-04535
                        if (layerCount<base_t::layers)
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-renderPass-04536
                        if (layerCount<=viewMaskMSB)
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00883
                        const auto levelCount = subresourceRange.levelCount!=ImageViewType::remaining_mip_levels ? subresourceRange.levelCount:imgParams.mipLevels;
                        if (levelCount!=1)
                            return true;

                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00884
                        if (viewParams.components!=ImageViewType::SComponentMapping())
                            return true;

                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-04533
                        if (imgParams.extent.width<base_t::width)
                            return true;
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-flags-04534
                        if (imgParams.extent.height<base_t::height)
                            return true;

                        const auto& desc = attachmentDesc[i];
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00880
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00881
                        if (viewParams.format!=desc.format || imgParams.samples!=desc.samples)
                            return true;

                        const auto usages = viewParams.actualUsages();
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00877
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-02633
                        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-02634
                        if (!usages.hasFlags(IImage::EUF_RENDER_ATTACHMENT_BIT))
                            return true;

                        if constexpr (std::is_same_v<decltype(desc),const IRenderpass::SCreationParams::SDepthStencilAttachmentDescription&>)
                        {
                            if (invalidUsageForLayout(desc.initialLayout.actualStencilLayout(),usages) || invalidUsageForLayout(desc.initialLayout.depth,usages))
                                return true;
                            if (invalidUsageForLayout(desc.finalLayout.actualStencilLayout(),usages) || invalidUsageForLayout(desc.finalLayout.depth,usages))
                                return true;

                            const auto viewType = viewParams.viewType;
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00891
                            if (imgParams.type==IImage::ET_3D && (viewType==ImageViewType::ET_2D||viewType==ImageViewType::ET_2D_ARRAY))
                                return true;
                        }
                        else
                        {
                            static_assert(std::is_same_v<decltype(desc),const IRenderpass::SCreationParams::SColorAttachmentDescription&>);
                            if (invalidUsageForLayout(desc.initialLayout,usages) || invalidUsageForLayout(desc.finalLayout,usages))
                                return true;
                        }
                    }
                    return false;
                };
                const auto& rpParams = rp->getCreationParameters();
                if (invalidAttachments(rp->getDepthStencilAttachmentCount(),rpParams.depthStencilAttachments,base_t::depthStencilAttachments))
                    return false;
                if (invalidAttachments(rp->getColorAttachmentCount(),rpParams.colorAttachments,base_t::colorAttachments))
                    return false;

                for (auto j=0u; j<rp->getSubpassCount(); j++)
                {
                    const IRenderpass::SCreationParams::SSubpassDescription& desc = rp->getCreationParameters().subpasses[j];
                    bool valid = true;
                    core::visit_token_terminated_array(desc.inputAttachments,IRenderpass::SCreationParams::SSubpassDescription::InputAttachmentsEnd,[&](const IRenderpass::SCreationParams::SSubpassDescription::SInputAttachmentRef& ia)->bool
                    {
                            const auto& viewParams = (ia.isColor() ? base_t::colorAttachments[ia.asColor.attachmentIndex]:base_t::depthStencilAttachments[ia.asDepthStencil.attachmentIndex])->getCreationParameters();
                            // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-pAttachments-00879
                            if (viewParams.actualUsages().hasFlags(IImage::EUF_INPUT_ATTACHMENT_BIT))
                                return (valid=false);
                            //TODO: https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkFramebufferCreateInfo.html#VUID-VkFramebufferCreateInfo-samples-07009
                            return true;
                    });
                    if (!valid)
                        return false;
                }

                return true;
            }
        };

        using SCachedCreationParams = SCreationParamsBase<false>;
        const SCachedCreationParams& getCreationParameters() const { return m_params; }

    protected:
        explicit IFramebuffer(const SCreationParams& params) : m_params{std::move(params.renderpass),nullptr,nullptr,params.width,params.height,params.layers}
        {
            const auto* const rp = m_params.renderpass.get();
            const auto depthStencilCount = rp->getDepthStencilAttachmentCount();
            const auto colorCount = rp->getColorAttachmentCount();
            m_attachments = core::make_refctd_dynamic_array<attachments_array_t>(depthStencilCount+colorCount);
            m_params.depthStencilAttachments = m_attachments->data();
            m_params.colorAttachments = m_params.depthStencilAttachments+depthStencilCount;
            for (auto i=0; i<depthStencilCount; i++)
                m_params.depthStencilAttachments[i] = core::smart_refctd_ptr<ImageViewType>(params.depthStencilAttachments[i]);
            for (auto i=0; i<colorCount; i++)
                m_params.colorAttachments[i] = core::smart_refctd_ptr<ImageViewType>(params.colorAttachments[i]);
        }
        virtual ~IFramebuffer() = default;

        SCachedCreationParams m_params;
        using attachments_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<ImageViewType>>;
        // storage for m_params.attachments
        attachments_array_t m_attachments;
};

}

#endif