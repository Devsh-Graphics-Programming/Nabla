#ifndef __NBL_I_FRAMBEBUFFER_H_INCLUDED__
#define __NBL_I_FRAMBEBUFFER_H_INCLUDED__

#include "nbl/core/IReferenceCounted.h"
#include "nbl/video/IGPUImageView.h"
#include "nbl/video/IGPURenderpass.h"

namespace nbl {
namespace video
{

class IFramebuffer : public core::IReferenceCounted
{
public:
    enum E_CREATE_FLAGS : uint8_t
    {
        ECF_IMAGELESS_BIT = 0x01
    };

    _NBL_STATIC_INLINE_CONSTEXPR uint32_t MaxAttachments = IGPURenderpass::EAP_COUNT;

    struct SCreationParams
    {
        E_CREATE_FLAGS flags;
        core::smart_refctd_ptr<IGPURenderpass> renderpass;
        core::smart_refctd_ptr<IGPUImageView> attachments[MaxAttachments];
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

        const uint32_t presentAttachments = std::count_if(std::begin(params.attachments), std::end(params.attachments), [](auto a) -> bool { return a; });

        if (rp->getAttachmentCount() != presentAttachments)
            return false;

        if (!(params.flags & ECF_IMAGELESS_BIT))
        {
            for (uint32_t i = 0u; i < MaxAttachments; ++i)
            {
                const auto a = static_cast<IGPURenderpass::E_ATTACHMENT_POINT>(i);

                if (static_cast<bool>(params.attachments[i]) != rp->isAttachmentEnabled(a))
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

            for (auto& a : params.attachments)
            {
                const auto& aParams = a->getCreationParameters();

                if (aParams.image->getCreationParameters().extent.height < params.height)
                    return false;
                if (aParams.subresourceRange.layerCount < params.layers)
                    return false;
                if (aParams.viewType == IGPUImageView::ET_3D)
                    return false;
            }
        }

        return true;
    }

    explicit IFramebuffer(SCreationParams&& params) : m_params(std::move(params))
    {

    }

protected:
    SCreationParams m_params;
};

}
}

#endif