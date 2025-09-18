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

   static SDualMeasureQuant<T> __call(const T microfacetMeasure, scalar_type clampedNdotV, scalar_type clampedNdotL, scalar_type VdotHLdotH, scalar_type VdotH_etaLdotH)
   {
      SDualMeasureQuant<T> retval;
      retval.microfacetMeasure = microfacetMeasure;
      // do constexpr booleans first so optimizer picks up this and short circuits
      const bool transmitted = reflect_refract==MTT_REFRACT || (reflect_refract!=MTT_REFLECT && VdotHLdotH < scalar_type(0.0));
      retval.projectedLightMeasure = microfacetMeasure*mix<scalar_type>(scalar_type(0.25),VdotHLdotH,transmitted);
      scalar_type denominator = clampedNdotV;
      if (transmitted) // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
      retval.projectedLightMeasure /= denominator;
      return retval;
   }
};
}

template<typename T>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, typename vector_traits<T>::scalar_type clampedNdotV, typename vector_traits<T>::scalar_type clampedNdotL)
{
   typename vector_traits<T>::scalar_type dummy;
   return impl::createDualMeasureQuantity_helper<T,MTT_REFLECT>::__call(specialMeasure,clampedNdotV,clampedNdotL,dummy,dummy);
}
template<typename T, MicrofacetTransformTypes reflect_refract>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, typename vector_traits<T>::scalar_type clampedNdotV, typename vector_traits<T>::scalar_type clampedNdotL, typename vector_traits<T>::scalar_type VdotHLdotH, typename vector_traits<T>::scalar_type VdotH_etaLdotH)
{
   return impl::createDualMeasureQuantity_helper<T,reflect_refract>::__call(specialMeasure,clampedNdotV,clampedNdotL,VdotHLdotH,VdotH_etaLdotH);
}

}
}
}
}

#endif
