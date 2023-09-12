#ifndef _NBL_I_GRAPHICS_PIPELINE_H_INCLUDED_
#define _NBL_I_GRAPHICS_PIPELINE_H_INCLUDED_

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
        core::smart_refctd_ptr<const renderpass_independent_t> renderpassIndependent;
        IImage::E_SAMPLE_COUNT_FLAGS rasterizationSamples = IImage::ESCF_1_BIT;
        core::smart_refctd_ptr<const RenderpassType> renderpass;
        uint32_t subpassIx = 0u;
        core::bitflag<ICPURenderpassIndependentPipeline::E_PIPELINE_CREATION> createFlags;
    };

    static inline bool validate(const SCreationParams& params)
    {
        const auto& rp = params.renderpass;
        if (params.subpassIx>=rp->getSubpassCount())
            return false;

        // TODO more validation

        return true;
    }

    explicit IGraphicsPipeline(SCreationParams&& _params) : m_params(std::move(_params))
    {

    }

    const renderpass_independent_t* getRenderpassIndependentPipeline() const { return m_params.renderpassIndependent.get(); }

    // TODO: why do we have these getters for lazy people!?
    const renderpass_t* getRenderpass() const { return m_params.renderpass.get(); }
    uint32_t getSubpassIndex() const { return m_params.subpassIx; }

    const SCreationParams& getCreationParameters() const { return m_params; }

protected:
    SCreationParams m_params;
};

}

#endif