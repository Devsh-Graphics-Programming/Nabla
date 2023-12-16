#ifndef _NBL_ASSET_I_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_GRAPHICS_PIPELINE_H_INCLUDED_

#include "nbl/asset/IRenderpassIndependentPipeline.h"
#include "nbl/asset/IRenderpass.h"

namespace nbl::asset
{

template<typename RenderpassIndependentType, typename RenderpassType>
class IGraphicsPipeline
{
    protected:
        using renderpass_t = RenderpassType;

    public:
        using renderpass_independent_t = RenderpassIndependentType;

        struct SCreationParams
        {
            inline bool valid() const
            {
                if (!renderpassIndependent || !renderpass)
                    return false;

                if (subpassIx>=renderpass->getSubpassCount())
                    return false;

                // TODO: check rasterization samples, etc.
                //rp->getCreationParameters().subpasses[i]
                return true;
            }

            core::smart_refctd_ptr<renderpass_independent_t> renderpassIndependent = nullptr;
            IImage::E_SAMPLE_COUNT_FLAGS rasterizationSamples = IImage::ESCF_1_BIT;
            core::smart_refctd_ptr<RenderpassType> renderpass = nullptr;
            uint32_t subpassIx = 0u;
        };

        const SCreationParams& getCreationParameters() const { return m_params; }

    protected:
        explicit IGraphicsPipeline(SCreationParams&& _params) : m_params(std::move(_params)) {}

        SCreationParams m_params;
};

}

#endif