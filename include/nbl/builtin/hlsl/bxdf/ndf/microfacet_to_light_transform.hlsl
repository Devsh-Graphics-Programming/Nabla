// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_MICROFACET_LIGHT_TRANSFORM_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_MICROFACET_LIGHT_TRANSFORM_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/common.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

enum MicrofacetTransformTypes : uint16_t
{
   MTT_REFLECT = 0b01,
   MTT_REFRACT = 0b10,
   MTT_REFLECT_REFRACT = 0b11
};

namespace microfacet_transform_concepts
{
#define NBL_CONCEPT_NAME QuantQuery
#define NBL_CONCEPT_TPLT_PRM_KINDS (typename)
#define NBL_CONCEPT_TPLT_PRM_NAMES (T)
#define NBL_CONCEPT_PARAM_0 (query, T)
NBL_CONCEPT_BEGIN(1)
#define query NBL_CONCEPT_PARAM_T NBL_CONCEPT_PARAM_0
NBL_CONCEPT_END(
    ((NBL_CONCEPT_REQ_TYPE)(T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getVdotHLdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
    ((NBL_CONCEPT_REQ_EXPR_RET_TYPE)((query.getNeg_rcp2_VdotH_etaLdotH()), ::nbl::hlsl::is_same_v, typename T::scalar_type))
);
#undef query
#include <nbl/builtin/hlsl/concepts/__end.hlsl>
}

template<typename T>
struct DualMeasureQuantQuery
{
   using scalar_type = T;

   // note in pbrt it's `abs(VdotH)*abs(LdotH)`
   // we leverage the fact that under transmission the sign must always be negative and rest of the code already accounts for that
   scalar_type getVdotHLdotH() NBL_CONST_MEMBER_FUNC { return VdotHLdotH; }
   scalar_type getNeg_rcp2_VdotH_etaLdotH () NBL_CONST_MEMBER_FUNC { return neg_rcp2_VdotH_etaLdotH ; }

   scalar_type VdotHLdotH;
   scalar_type neg_rcp2_VdotH_etaLdotH;
};


template<typename T>
struct SDualMeasureQuant
{
   using value_type = T;
   
   T microfacetMeasure;
   T projectedLightMeasure;
};

namespace impl
{
template<typename T, MicrofacetTransformTypes reflect_refract>
struct createDualMeasureQuantity_helper
{
   using scalar_type = typename vector_traits<T>::scalar_type;

   static SDualMeasureQuant<T> __call(const T microfacetMeasure, scalar_type clampedNdotV, scalar_type clampedNdotL, scalar_type VdotHLdotH, scalar_type neg_rcp2_VdotH_etaLdotH)
   {
      assert(clampedNdotV >= scalar_type(0.0) && clampedNdotL >= scalar_type(0.0));
      SDualMeasureQuant<T> retval;
      retval.microfacetMeasure = microfacetMeasure;
      // do constexpr booleans first so optimizer picks up this and short circuits
      const bool transmitted = reflect_refract==MTT_REFRACT || (reflect_refract!=MTT_REFLECT && VdotHLdotH < scalar_type(0.0));
      retval.projectedLightMeasure = microfacetMeasure * hlsl::mix(scalar_type(0.25),VdotHLdotH*neg_rcp2_VdotH_etaLdotH,transmitted)/clampedNdotV;
      // VdotHLdotH is negative under transmission, so thats denominator is negative
      return retval;
   }
};
}

// specialMeasure meaning the measure defined by the specialization of createDualMeasureQuantity_helper; note that GGX redefines it somewhat
template<typename T>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, typename vector_traits<T>::scalar_type clampedNdotV, typename vector_traits<T>::scalar_type clampedNdotL)
{
   typename vector_traits<T>::scalar_type dummy;
   return impl::createDualMeasureQuantity_helper<T,MTT_REFLECT>::__call(specialMeasure,clampedNdotV,clampedNdotL,dummy,dummy);
}
template<typename T, MicrofacetTransformTypes reflect_refract>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, typename vector_traits<T>::scalar_type clampedNdotV, typename vector_traits<T>::scalar_type clampedNdotL, typename vector_traits<T>::scalar_type VdotHLdotH, typename vector_traits<T>::scalar_type neg_rcp2_VdotH_etaLdotH)
{
   return impl::createDualMeasureQuantity_helper<T,reflect_refract>::__call(specialMeasure,clampedNdotV,clampedNdotL,VdotHLdotH,neg_rcp2_VdotH_etaLdotH);
}
template<typename T, MicrofacetTransformTypes reflect_refract, class Query>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, typename vector_traits<T>::scalar_type clampedNdotV, typename vector_traits<T>::scalar_type clampedNdotL, NBL_CONST_REF_ARG(Query) query)
{
   return impl::createDualMeasureQuantity_helper<T,reflect_refract>::__call(specialMeasure,clampedNdotV,clampedNdotL,query.getVdotHLdotH(),query.getNeg_rcp2_VdotH_etaLdotH());
}

}
}
}
}

#endif
