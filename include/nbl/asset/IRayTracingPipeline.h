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
    #define base_flag(F) static_cast<uint64_t>(IPipelineBase::FLAGS::F)
    enum class CreationFlags : uint64_t
    {
      NONE = base_flag(NONE),
      DISABLE_OPTIMIZATIONS = base_flag(DISABLE_OPTIMIZATIONS),
      ALLOW_DERIVATIVES = base_flag(ALLOW_DERIVATIVES),
      FAIL_ON_PIPELINE_COMPILE_REQUIRED = base_flag(FAIL_ON_PIPELINE_COMPILE_REQUIRED),
      EARLY_RETURN_ON_FAILURE = base_flag(EARLY_RETURN_ON_FAILURE),
      SKIP_BUILT_IN_PRIMITIVES = 1<<12,
      SKIP_AABBS = 1<<13,
      NO_NULL_ANY_HIT_SHADERS = 1<<14,
      NO_NULL_CLOSEST_HIT_SHADERS = 1<<15,
      NO_NULL_MISS_SHADERS = 1<<16,
      NO_NULL_INTERSECTION_SHADERS = 1<<17,
      ALLOW_MOTION = 1<<20,
    };
    #undef base_flag

    struct SCachedCreationParams final
    {
      core::bitflag<CreationFlags> flags = CreationFlags::NONE;
      uint32_t maxRecursionDepth : 6 = 0;
      uint32_t dynamicStackSize : 1 = false;
    };
};

template<typename PipelineLayoutType>
class IRayTracingPipeline : public IPipeline<PipelineLayoutType>, public IRayTracingPipelineBase
{
  public:

    inline const SCachedCreationParams& getCachedCreationParams() const { return m_params; }

  protected:
    explicit IRayTracingPipeline(PipelineLayoutType* layout, const SCachedCreationParams& cachedParams) :
        IPipeline<PipelineLayoutType>(core::smart_refctd_ptr<PipelineLayoutType>(layout)),
        m_params(cachedParams)
    {}

    SCachedCreationParams m_params;

};

}

#endif
