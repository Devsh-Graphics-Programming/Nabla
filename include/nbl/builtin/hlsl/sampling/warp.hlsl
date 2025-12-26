#ifndef _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_WARP_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_CONCEPTS_WARP_INCLUDED_


namespace nbl
{
namespace hlsl
{
namespace sampling
{
  
template <typename CodomainT>
struct WarpResult
{
  CodomainT dst;
  float32_t density;
};
}

namespace concepts
{

// declare concept
#define NBL_CONCEPT_NAME Warp
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (U)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (warper,U)
#define NBL_CONCEPT_PARAM_1 (xi,typename U::domain_type)
#define NBL_CONCEPT_PARAM_2 (dst,typename U::codomain_type)
// start concept
NBL_CONCEPT_BEGIN(3)
#define warper NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define xi NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define dst NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(U::domain_type))
);
#undef dst
#undef xi
#undef warper
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}

}
}

#endif