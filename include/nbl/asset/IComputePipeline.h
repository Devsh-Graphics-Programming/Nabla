#ifndef _NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED_

#include "nbl/asset/IPipeline.h"

namespace nbl::asset
{

class IComputePipelineBase : public virtual core::IReferenceCounted
{
  public:

    struct SCachedCreationParams final
    {
        uint8_t requireFullSubgroups = false;
    };
};

template<typename PipelineLayoutType>
class IComputePipeline : public IPipeline<PipelineLayoutType>, public IComputePipelineBase
{
    using base_creation_params_t = IPipeline<PipelineLayoutType>;

  public:

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }

  protected:
    explicit IComputePipeline(const PipelineLayoutType* layout, const SCachedCreationParams& cachedParams) :
        IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<const PipelineLayoutType>(layout)),
        m_params(cachedParams)
    {}

    SCachedCreationParams m_params;

};

}

#endif
