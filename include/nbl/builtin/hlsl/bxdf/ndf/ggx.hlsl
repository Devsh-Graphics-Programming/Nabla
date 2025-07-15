// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_GGX_INCLUDED_

#include "nbl/builtin/hlsl/limits.hlsl"
#include "nbl/builtin/hlsl/bxdf/ndf.hlsl"

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

// TODO: use query_type when that's implemented
template<typename T>
struct GGX<T,false>
{
    using scalar_type = T;
    using this_t = GGX<T,false>;

    // trowbridge-reitz
    scalar_type D(scalar_type a2, scalar_type NdotH2)
    {
        scalar_type denom = NdotH2 * (a2 - 1.0) + 1.0;
        return a2 * numbers::inv_pi<scalar_type> / (denom * denom);
    }

    scalar_type FVNDF(scalar_type fresnel_ndf, scalar_type G1_over_2NdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta)
    {
        scalar_type FNG = fresnel_ndf * G1_over_2NdotV;
        scalar_type factor = 0.5;
        if (transmitted)
        {
            const scalar_type VdotH_etaLdotH = (VdotH + orientedEta * LdotH);
            // VdotHLdotH is negative under transmission, so this factor is negative
            factor *= -2.0 * VdotHLdotH / (VdotH_etaLdotH * VdotH_etaLdotH);
        }
        return FNG * factor;
    }

    scalar_type DG1(scalar_type ndf, scalar_type G1_over_2NdotV)
    {
        return ndf * 0.5 * G1_over_2NdotV;
    }

    scalar_type DG1(scalar_type ndf, scalar_type G1_over_2NdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta, scalar_type reflectance)
    {
        scalar_type FN = hlsl::mix(reflectance, scalar_type(1.0) - reflectance, transmitted) * ndf;
        return FVNDF(FN, G1_over_2NdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
    }

    scalar_type devsh_part(scalar_type NdotX2, scalar_type a2, scalar_type one_minus_a2)
    {
        return sqrt(a2 + one_minus_a2 * NdotX2);
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type NdotX2, scalar_type a2, scalar_type one_minus_a2)
    {
        return 1.0 / (NdotX + devsh_part(NdotX2,a2,one_minus_a2));
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type devsh_part)
    {
        return 1.0 / (NdotX + devsh_part);
    }

    scalar_type correlated_wo_numerator(scalar_type a2, scalar_type NdotV, scalar_type NdotV2, scalar_type NdotL, scalar_type NdotL2)
    {
        scalar_type one_minus_a2 = scalar_type(1.0) - a2;
        scalar_type Vterm = NdotL * devsh_part(NdotV2, a2, one_minus_a2);
        scalar_type Lterm = NdotV * devsh_part(NdotL2, a2, one_minus_a2);
        return 0.5 / (Vterm + Lterm);
    }

    scalar_type G2_over_G1(scalar_type a2, scalar_type NdotV, scalar_type NdotV2, scalar_type NdotL, scalar_type NdotL2)
    {
        scalar_type one_minus_a2 = scalar_type(1.0) - a2;
        scalar_type devsh_v = devsh_part(NdotV2, a2, one_minus_a2);
        scalar_type G2_over_G1 = NdotL * (devsh_v + NdotV); // alternative `Vterm+NdotL*NdotV /// NdotL*NdotV could come as a parameter
        G2_over_G1 /= NdotV * devsh_part(NdotL2, a2, one_minus_a2) + NdotL * devsh_v;

        return G2_over_G1;
    }
};

template<typename T>
struct GGX<T,true>
{
    using scalar_type = T;

    scalar_type D(scalar_type ax, scalar_type ay, scalar_type ax2, scalar_type ay2, scalar_type TdotH2, scalar_type BdotH2, scalar_type NdotH2)
    {
        scalar_type a2 = ax * ay;
        scalar_type denom = TdotH2 / ax2 + BdotH2 / ay2 + NdotH2;
        return numbers::inv_pi<scalar_type> / (a2 * denom * denom);
    }

    // burley
    scalar_type D(scalar_type a2, scalar_type TdotH, scalar_type BdotH, scalar_type NdotH, scalar_type anisotropy)
    {
        scalar_type antiAniso = 1.0 - anisotropy;
        scalar_type atab = a2 * antiAniso;
        scalar_type anisoTdotH = antiAniso * TdotH;
        scalar_type anisoNdotH = antiAniso * NdotH;
        scalar_type w2 = antiAniso/(BdotH * BdotH + anisoTdotH * anisoTdotH + anisoNdotH * anisoNdotH * a2);
        return w2 * w2 * atab * numbers::inv_pi<scalar_type>;
    }

    scalar_type DG1(scalar_type ndf, scalar_type G1_over_2NdotV)
    {
        GGX<T,false> ggx;
        return ggx.DG1(ndf, G1_over_2NdotV);
    }

    scalar_type DG1(scalar_type ndf, scalar_type G1_over_2NdotV, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta, scalar_type reflectance)
    {
        GGX<T,false> ggx;
        return ggx.DG1(ndf, G1_over_2NdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta, reflectance);
    }

    scalar_type devsh_part(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return sqrt(TdotX2 * ax2 + BdotX2 * ay2 + NdotX2);
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return 1.0 / (NdotX + devsh_part(TdotX2, BdotX2, NdotX2, ax2, ay2));
    }

    scalar_type G1_wo_numerator(scalar_type NdotX, scalar_type devsh_part)
    {
        return 1.0 / (NdotX + devsh_part);
    }

    scalar_type correlated_wo_numerator(scalar_type ax2, scalar_type ay2, scalar_type NdotV, scalar_type TdotV2, scalar_type BdotV2, scalar_type NdotV2, scalar_type NdotL, scalar_type TdotL2, scalar_type BdotL2, scalar_type NdotL2)
    {
        scalar_type Vterm = NdotL * devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
        scalar_type Lterm = NdotV * devsh_part(TdotL2, BdotL2, NdotL2, ax2, ay2);
        return 0.5 / (Vterm + Lterm);
    }

    scalar_type G2_over_G1(scalar_type ax2, scalar_type ay2, scalar_type NdotV, scalar_type TdotV2, scalar_type BdotV2, scalar_type NdotV2, scalar_type NdotL, scalar_type TdotL2, scalar_type BdotL2, scalar_type NdotL2)
    {
        scalar_type devsh_v = devsh_part(TdotV2, BdotV2, NdotV2, ax2, ay2);
        scalar_type G2_over_G1 = NdotL * (devsh_v + NdotV);
        G2_over_G1 /= NdotV * devsh_part(TdotL2, BdotL2, NdotL2, ax2, ay2) + NdotL * devsh_v;

        return G2_over_G1;
    }
};

}
}
}
}

#endif
