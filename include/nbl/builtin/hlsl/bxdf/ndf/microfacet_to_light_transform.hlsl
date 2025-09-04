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
   using scalar_type = vector_traits<T>::scalar_type;

   static SDualMeasureQuant<T> __call(const T microfacetMeasure, scalar_type clampedNdotV, scalar_type clampedNdotL, scalar_type VdotHLdotH, scalar_type VdotH_etaLdotH)
   {
      SDualMeasureQuant<T> retval;
      retval.microfacetMeasure = microfacetMeasure;
      // do constexpr booleans first so optimizer picks up this and short circuits
      const bool transmitted = reflect_refract==MTT_REFRACT ||  reflect_refract!=MTT_REFLECT && VdotHLdotH<scalar_type(0);
      retval.projectedLightMeasure = microfacetMeasure*mix<scalar_type>(0.25,VdotHLdotH,transmitted);
      scalar_type denominator = clampedNdotV;
      if (transmitted) // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
      retval.projectedLightMeasure /= denominator;
      return retval;
   }
};
}

template<typename T>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, vector_traits<T>::scalar_type clampedNdotV, vector_traits<T>::scalar_type clampedNdotL)
{
   vector_traits<T>::scalar_type dummy;
   return impl::createDualMeasureQuantity_helper<T,MTT_REFLECT>::__call(specialMeasure,clampedNdotV,clampedNdotL,dummy,dummy);
}
template<typename T, MicrofacetTransformTypes reflect_refract>
SDualMeasureQuant<T> createDualMeasureQuantity(const T specialMeasure, vector_traits<T>::scalar_type clampedNdotV, vector_traits<T>::scalar_type clampedNdotL, vector_traits<T>::scalar_type VdotHLdotH, vector_traits<T>::scalar_type VdotH_etaLdotH)
{
   return impl::createDualMeasureQuantity_helper<T,reflect_refract>::__call(specialMeasure,clampedNdotV,clampedNdotL,VdotHLdotH,VdotH_etaLdotH);
}

/*
template<typename T, bool IsGgx, MicrofacetTransformTypes reflect_refract>
struct SDualMeasureQuant;

template<typename T>
struct SDualMeasureQuant<T, false, MTT_REFLECT>
{
    using this_t = SDualMeasureQuant<T, false, MTT_REFLECT>;
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE MicrofacetTransformTypes Type = MTT_REFLECT;

    scalar_type getMicrofacetMeasure()
    {
        return pdf;
    }

    // this computes the max(NdotL,0)/(4*max(NdotV,0)*max(NdotL,0)) factor which transforms PDFs in the f in projected microfacet f * NdotH measure to projected light measure f * NdotL
    scalar_type getProjectedLightMeasure()
    {
        return scalar_type(0.25) * pdf / maxNdotV;
    }

    scalar_type pdf;
    scalar_type maxNdotV;
};

template<typename T>
struct SDualMeasureQuant<T, true, MTT_REFLECT>
{
    using this_t = SDualMeasureQuant<T, true, MTT_REFLECT>;
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE MicrofacetTransformTypes Type = MTT_REFLECT;

    scalar_type getMicrofacetMeasure()
    {
        return pdf;
    }

    // this computes the max(NdotL,0)/(4*max(NdotV,0)*max(NdotL,0)) factor which transforms PDFs in the f in projected microfacet f * NdotH measure to projected light measure f * NdotL
    scalar_type getProjectedLightMeasure()
    {
        return pdf * maxNdotL;
    }

    scalar_type pdf;
    scalar_type maxNdotL;
};

template<typename T>
struct SDualMeasureQuant<T, false, MTT_REFRACT>
{
    using this_t = SDualMeasureQuant<T, false, MTT_REFRACT>;
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE MicrofacetTransformTypes Type = MTT_REFRACT;

    scalar_type getMicrofacetMeasure()
    {
        return pdf;
    }

    scalar_type getProjectedLightMeasure()
    {
        const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        scalar_type denominator = absNdotV * (-VdotH_etaLdotH * VdotH_etaLdotH);
        return pdf * VdotHLdotH / denominator;
    }

    scalar_type pdf;
    scalar_type absNdotV;
    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type VdotHLdotH;
    scalar_type orientedEta;
};

template<typename T>
struct SDualMeasureQuant<T, true, MTT_REFRACT>
{
    using this_t = SDualMeasureQuant<T, true, MTT_REFRACT>;
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE MicrofacetTransformTypes Type = MTT_REFRACT;

    scalar_type getMicrofacetMeasure()
    {
        return pdf;
    }

    scalar_type getProjectedLightMeasure()
    {
        const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
        // VdotHLdotH is negative under transmission, so thats denominator is negative
        scalar_type denominator = absNdotL * (-scalar_type(4.0) * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH));
        return pdf * denominator;
    }

    scalar_type pdf;
    scalar_type absNdotL;
    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type VdotHLdotH;
    scalar_type orientedEta;
};

template<typename T>
struct SDualMeasureQuant<T, false, MTT_REFLECT_REFRACT>
{
    using this_t = SDualMeasureQuant<T, false, MTT_REFLECT_REFRACT>;
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE MicrofacetTransformTypes Type = MTT_REFLECT_REFRACT;

    scalar_type getMicrofacetMeasure()
    {
        return pdf;
    }

    scalar_type getProjectedLightMeasure()
    {
        scalar_type denominator = absNdotV;
        if (transmitted)
        {
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -VdotH_etaLdotH * VdotH_etaLdotH;
        }
        return pdf * (transmitted ? VdotHLdotH : scalar_type(0.25)) / denominator;
    }

    scalar_type pdf;
    scalar_type absNdotV;
    bool transmitted;
    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type VdotHLdotH;
    scalar_type orientedEta;
};

template<typename T>
struct SDualMeasureQuant<T, true, MTT_REFLECT_REFRACT>
{
    using this_t = SDualMeasureQuant<T, true, MTT_REFLECT_REFRACT>;
    using scalar_type = T;

    NBL_CONSTEXPR_STATIC_INLINE MicrofacetTransformTypes Type = MTT_REFLECT_REFRACT;

    scalar_type getMicrofacetMeasure()
    {
        return pdf;
    }

    scalar_type getProjectedLightMeasure()
    {
        scalar_type denominator = absNdotL;
        if (transmitted)
        {
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so thats denominator is negative
            denominator *= -scalar_type(4.0) * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
        }
        return pdf * denominator;
    }

    scalar_type pdf;
    scalar_type absNdotL;
    bool transmitted;
    scalar_type VdotH;
    scalar_type LdotH;
    scalar_type VdotHLdotH;
    scalar_type orientedEta;
};
*/

}
}
}
}

#endif
