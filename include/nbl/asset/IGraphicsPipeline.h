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
    using renderpass_independent_t = RenderpassIndependentType;
    using renderpass_t = RenderpassType;

public:
    struct SCreationParams
    {
        core::smart_refctd_ptr<const renderpass_independent_t> renderpassIndependent;
        IImage::E_SAMPLE_COUNT_FLAGS rasterizationSamplesHint = IImage::ESCF_1_BIT;
        core::smart_refctd_ptr<const RenderpassType> renderpass;
        uint32_t subpassIx = 0u;
    };

    static bool validate(const SCreationParams& params)
    {
        // TODO more validation

        auto& rp = params.renderpass;
        uint32_t sp = params.subpassIx;
        if (sp >= rp->getSubpasses().size())
            return false;
        return true;
    }

    explicit IGraphicsPipeline(SCreationParams&& _params) : m_params(std::move(_params))
    {

    }

    const renderpass_independent_t* getRenderpassIndependentPipeline() const { return m_params.renderpassIndependent.get(); }
    const renderpass_t* getRenderpass() const { return m_params.renderpass.get(); }
    uint32_t getSubpassIndex() const { return m_params.subpassIx; }
    IImage::E_SAMPLE_COUNT_FLAGS getRasterSamplesHint() const { return m_params.rasterizationSamplesHint; }
    const SCreationParams& getCreationParameters() const { return m_params; }

protected:
    SCreationParams m_params;
};

}

#endif