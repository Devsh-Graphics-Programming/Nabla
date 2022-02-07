#ifndef __NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED__
#define __NBL_I_CPU_GRAPHICS_PIPELINE_H_INCLUDED__

#include "nbl/asset/IGraphicsPipeline.h"
#include "nbl/asset/IAsset.h"
#include "nbl/asset/ICPURenderpassIndependentPipeline.h"
#include "nbl/asset/ICPURenderpass.h"

namespace nbl
{
namespace asset
{
class ICPUGraphicsPipeline final : public IAsset, public IGraphicsPipeline<ICPURenderpassIndependentPipeline, ICPURenderpass>
{
    using base_t = IGraphicsPipeline<ICPURenderpassIndependentPipeline, ICPURenderpass>;

public:
    ~ICPUGraphicsPipeline()
    {
    }

    using base_t::base_t;

    renderpass_independent_t* getRenderpassIndependentPipeline() { return m_params.renderpassIndependent.get(); }
    renderpass_t* getRenderpass() { return m_params.renderpass.get(); }
    SCreationParams& getCreationParameters() { return m_params; }
};

}
}

#endif