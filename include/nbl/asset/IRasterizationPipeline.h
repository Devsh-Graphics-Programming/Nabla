#ifndef _NBL_ASSET_I_RASTERIZATION_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RASTERIZATION_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/RasterizationStates.h"
#include "nbl/asset/IPipeline.h"
#include "nbl/asset/IRenderpass.h"


//the primary goal is to abstract between mesh and traditional graphics (vertex) pipelines
//so that any pipeline that can be bound to VK_PIPELINE_BIND_POINT_GRAPHICS can be returned polymorphically
//the secondary goal is to not change IGraphicsPipeline as little as possible

namespace nbl::asset {

class IRasterizationPipelineBase
{
    //IRasterizationPipelineBase isn't inherited from, only SCachedCreationParams 
    public:
        struct SCachedCreationParams {
            SRasterizationParams rasterization = {};
            SBlendParams blend = {};
            uint32_t subpassIx = 0u;
        };
};

template<typename PipelineLayoutType, typename RenderpassType>
class IRasterizationPipeline : public IPipeline<PipelineLayoutType>
{
protected:
    using renderpass_t = RenderpassType;

public:
    inline const renderpass_t* getRenderpass() const { return m_renderpass.get(); }
protected:
    explicit IRasterizationPipeline(PipelineLayoutType* layout, renderpass_t* renderpass) :
        IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<PipelineLayoutType>(layout)),
        m_renderpass(core::smart_refctd_ptr<renderpass_t>(renderpass))
    {}

    core::smart_refctd_ptr<renderpass_t> m_renderpass = nullptr;
};
}


#endif