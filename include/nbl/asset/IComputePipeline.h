#ifndef _NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED_
#define _NBL_ASSET_I_COMPUTE_PIPELINE_H_INCLUDED_

#include "nbl/asset/IPipeline.h"

namespace nbl::asset
{

class IComputePipelineBase : public virtual core::IReferenceCounted
{
  public:
    // Nabla requires device's reported subgroup size to be between 4 and 128
    enum class SUBGROUP_SIZE : uint8_t
    {
      // No constraint but probably means `gl_SubgroupSize` is Dynamically Uniform
      UNKNOWN = 0,
      // Allows the Subgroup Uniform `gl_SubgroupSize` to be non-Dynamically Uniform and vary between Device's min and max
      VARYING = 1,
      // The rest we encode as log2(x) of the required value
      REQUIRE_4 = 2,
      REQUIRE_8 = 3,
      REQUIRE_16 = 4,
      REQUIRE_32 = 5,
      REQUIRE_64 = 6,
      REQUIRE_128 = 7
    };

    struct SCachedCreationParams final
    {
        SUBGROUP_SIZE requiredSubgroupSize : 3 = SUBGROUP_SIZE::UNKNOWN;	//!< Default value of 8 means no requirement
        uint8_t requireFullSubgroups : 1 = false;
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
