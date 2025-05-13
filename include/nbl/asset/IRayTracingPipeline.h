#ifndef _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_RAY_TRACING_PIPELINE_H_INCLUDED_

#include "nbl/asset/IShader.h"
#include "nbl/asset/IPipeline.h"

#include <span>
#include <bit>
#include <type_traits>

namespace nbl::asset
{

class IRayTracingPipelineBase : public virtual core::IReferenceCounted
{
  public:
    struct SCachedCreationParams final
    {
      uint32_t maxRecursionDepth : 6 = 0;
      uint32_t dynamicStackSize : 1 = false;
    };
};

template<typename PipelineLayoutType>
class IRayTracingPipeline : public IPipeline<PipelineLayoutType>, public IRayTracingPipelineBase
{
    using base_creation_params_t = IPipeline<PipelineLayoutType>;

  public:

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }

  protected:
    explicit IRayTracingPipeline(const PipelineLayoutType* layout, const SCachedCreationParams& cachedParams) :
        IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<const PipelineLayoutType>(layout)),
        m_params(cachedParams)
    {}

    SCachedCreationParams m_params;

};

}

#endif
