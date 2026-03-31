#ifndef _NBL_BUILTIN_HLSL_HIERARCHICAL_IMAGE_ACCESSORS_INCLUDED_
#define _NBL_BUILTIN_HLSL_CONCEPTS_ACCESSORS_HIERARCHICAL_IMAGE_INCLUDED_

#include "nbl/builtin/hlsl/concepts/accessors/generic_shared_data.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{
namespace hierarchical_image
{

// declare concept
#define NBL_CONCEPT_NAME MipmappedLuminanceReadAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (AccessorT)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (accessor,AccessorT)
#define NBL_CONCEPT_PARAM_1 (pixelCoord,uint16_t2)
#define NBL_CONCEPT_PARAM_2 (level,uint16_t)
#define NBL_CONCEPT_PARAM_3 (outVal,typename AccessorT::value_type)
// start concept
NBL_CONCEPT_BEGIN(4)
// need to be defined AFTER the concept begins
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define pixelCoord NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define level NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define outVal NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(AccessorT::value_type))
    // Note(kevin): I don't use MipmappedLoadableImage here, since that concept require layer as parameter. So the sampler have to store the layerIndex. The logic is similar across all layer. So the accessor should be the one that store the layerIndex
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.template get(outVal,pixelCoord,level)) , ::nbl::hlsl::is_same_v, void))
    // Ask(kevin): Should getAvgLuma follow get, where the outVal is the first parameter instead of the return value?
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.getAvgLuma()), ::nbl::hlsl::is_same_v, typename AccessorT::value_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.resolution()), ::nbl::hlsl::is_same_v, uint16_t2))
);
#undef accessor
#undef pixelCoord
#undef level
#undef outVal
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

// declare concept
#define NBL_CONCEPT_NAME LuminanceReadAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (AccessorT)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (accessor, AccessorT)
// start concept
NBL_CONCEPT_BEGIN(1)
// need to be defined AFTER the concept begins
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(AccessorT::value_type))
    ((NBL_CONCEPT_REQ_TYPE_ALIAS_CONCEPT)(concepts::accessors::GenericReadAccessor, AccessorT, typename AccessorT::value_type, float32_t2))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.getAvgLuma()), ::nbl::hlsl::is_same_v, typename AccessorT::value_type))
);
#undef accessor
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

// gatherUvs return 4 UVs in a square for manual bilinear interpolation with differentiability
// declare concept
#define NBL_CONCEPT_NAME WarpAccessor
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (WarpAccessorT)
// not the greatest syntax but works
#define NBL_CONCEPT_PARAM_0 (accessor,WarpAccessorT)
#define NBL_CONCEPT_PARAM_1 (coord,vector<float32_t, 2>)
#define NBL_CONCEPT_PARAM_2 (val, matrix<typename WarpAccessorT::scalar_type, 4, 2>)
#define NBL_CONCEPT_PARAM_3 (interpolant, vector<typename WarpAccessorT::scalar_type, 2>)
// start concept
NBL_CONCEPT_BEGIN(4)
// need to be defined AFTER the concept begins
#define accessor NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
#define coord NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_1
#define val NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_2
#define interpolant NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_3
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(WarpAccessorT::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.gatherUv(coord, val)), ::nbl::hlsl::is_same_v, void))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((accessor.resolution()), ::nbl::hlsl::is_same_v, uint16_t2))
);
#undef accessor
#undef coord
#undef val
#undef interpolant
#include <nbl/builtin/hlsl/concepts/__end.hlsl>

}
}
}
}

#endif
