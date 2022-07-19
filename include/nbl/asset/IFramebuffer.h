#ifndef __NBL_I_FRAMBEBUFFER_H_INCLUDED__
#define __NBL_I_FRAMBEBUFFER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPURenderpass.h"

namespace nbl {
namespace asset
{

template <typename RenderpassType, typename ImageViewType>
class NBL_API IFramebuffer
{
public:
    using renderpass_t = RenderpassType;
    using attachment_t = ImageViewType;

    enum E_CREATE_FLAGS : uint8_t
    {
        ECF_IMAGELESS_BIT = 0x01
    };

    struct SCreationParams
    {
        E_CREATE_FLAGS flags;
        core::smart_refctd_ptr<renderpass_t> renderpass;
        uint32_t attachmentCount;
        core::smart_refctd_ptr<attachment_t>* attachments;
        uint32_t width;
        uint32_t height;
        uint32_t layers;
    };

    static bool validate(const SCreationParams& params)
    {
        if (!params.width)
            return false;
        if (!params.height)
            return false;
        if (!params.layers)
            return false;

        auto* rp = params.renderpass.get();

        const uint32_t presentAttachments = params.attachmentCount;

        if (rp->getCreationParameters().attachmentCount != presentAttachments)
            return false;

        if (!(params.flags & ECF_IMAGELESS_BIT))
        {
            for (uint32_t i = 0u; i < params.attachmentCount; ++i)
            {
                auto& a = params.attachments[i];
                const asset::E_FORMAT this_format = a->getCreationParameters().format;
                const asset::E_FORMAT rp_format = rp->getAttachments().begin()[i].format;

                if (this_format != rp_format)
                    return false;
            }

            /*
            * TODO
            * when we have E_USAGE_FLAGS in IImage:
            * 
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a color 
            attachment or resolve attachment by renderPass must have been created with a usage value including VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as a depth/stencil attachment 
            by renderPass must have been created with a usage value including VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
            * If flags does not include VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT, each element of pAttachments that is used as an input attachment
            by renderPass must have been created with a usage value including VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
            */

            auto attachments = core::SRange<core::smart_refctd_ptr<attachment_t>>{params.attachments, params.attachments + params.attachmentCount};
            for (auto& a : attachments)
            {
                const auto& aParams = a->getCreationParameters();

                if (aParams.image->getCreationParameters().extent.width < params.width)
                    return false;
                if (aParams.image->getCreationParameters().extent.height < params.height)
                    return false;
                if (aParams.subresourceRange.layerCount < params.layers)
                    return false;
                if (aParams.subresourceRange.levelCount != 1u)
                    return false;
                if (aParams.viewType == ImageViewType::ET_3D)
                    return false;
            }
        }

        return true;
    }

    const SCreationParams& getCreationParameters() const { return m_params; }

protected:
    explicit IFramebuffer(SCreationParams&& params) : m_params(std::move(params))
    {
        if (m_params.flags & ECF_IMAGELESS_BIT)
        {
            m_params.attachments = nullptr;
            m_params.attachmentCount = 0u;
        }
        else
        {
            auto attachments = core::SRange<core::smart_refctd_ptr<attachment_t>>{m_params.attachments, m_params.attachments + m_params.attachmentCount};

            m_attachments = core::make_refctd_dynamic_array<attachments_array_t>(m_params.attachmentCount);
            std::copy(attachments.begin(), attachments.end(), m_attachments->begin());
            m_params.attachments = m_attachments->data();
        }
    }
    virtual ~IFramebuffer() = default;

    SCreationParams m_params;
    using attachments_array_t = core::smart_refctd_dynamic_array<core::smart_refctd_ptr<attachment_t>>;
    // storage for m_params.attachments
    attachments_array_t m_attachments;
};

}
}

#endif