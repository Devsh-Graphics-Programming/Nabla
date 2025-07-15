// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_BECKMANN_INCLUDED_

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

// TODO: get beta from lgamma, see: https://www.cec.uchile.cl/cinetica/pcordero/MC_libros/NumericalRecipesinC.pdf

// TODO: use query_type for D, lambda, beta, DG1 when that's implemented
template<typename T>
struct Beckmann<T,false>
{
    using scalar_type = T;
    using this_t = Beckmann<T,false>;

    scalar_type D(scalar_type a2, scalar_type NdotH2)
    {
        scalar_type nom = exp<scalar_type>( (NdotH2 - 1.0) / (a2 * NdotH2) );   // exp(x) == exp2(x/log(2)) ?
        scalar_type denom = a2 * NdotH2 * NdotH2;
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    // brdf
    scalar_type DG1(scalar_type ndf, scalar_type maxNdotV, scalar_type lambda_V)
    {
        onePlusLambda_V = 1.0 + lambda_V;
        return ndf::microfacet_to_light_measure_transform<this_t,ndf::MTT_REFLECT>::__call(ndf / onePlusLambda_V, maxNdotV);
    }

    // bsdf
    scalar_type DG1(scalar_type ndf, scalar_type absNdotV, scalar_type lambda_V, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta, scalar_type reflectance)
    {
        onePlusLambda_V = 1.0 + lambda_V;
        return ndf::microfacet_to_light_measure_transform<this_t,ndf::MTT_REFLECT_REFRACT>::__call(hlsl::mix(reflectance, scalar_type(1.0) - reflectance, transmitted) * ndf / onePlusLambda_V, absNdotV, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta);
    }

    scalar_type G1(scalar_type lambda)
    {
        return 1.0 / (1.0 + lambda);
    }

    scalar_type C2(scalar_type NdotX2, scalar_type a2)
    {
        return NdotX2 / (a2 * (1.0 - NdotX2));
    }

    scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = 1.0 - 1.259 * c + 0.396 * c2;
        scalar_type denom = 2.181 * c2 + 3.535 * c;
        return hlsl::mix<scalar_type>(0.0, nom / denom, c < 1.6);
    }

    scalar_type Lambda(scalar_type NdotX2, scalar_type a2)
    {
        return Lambda(C2(NdotX2, a2));
    }

    scalar_type correlated(scalar_type a2, scalar_type NdotV2, scalar_type NdotL2)
    {
        scalar_type c2 = C2(NdotV2, a2);
        scalar_type L_v = Lambda(c2);
        c2 = C2(NdotL2, a2);
        scalar_type L_l = Lambda(c2);
        return G1(L_v + L_l);
    }

    scalar_type G2_over_G1(scalar_type a2, scalar_type NdotL2, scalar_type lambdaV_plus_one)
    {
        scalar_type lambdaL = Lambda(NdotL2, a2);
        return lambdaV_plus_one / (lambdaV_plus_one + lambdaL);
    }

    scalar_type onePlusLambda_V;
};


template<typename T>
struct Beckmann<T,true>
{
    using scalar_type = T;

    scalar_type D(scalar_type ax, scalar_type ay, scalar_type ax2, scalar_type ay2, scalar_type TdotH2, scalar_type BdotH2, scalar_type NdotH2)
    {
        scalar_type nom = exp<scalar_type>(-(TdotH2 / ax2 + BdotH2 / ay2) / NdotH2);
        scalar_type denom = ax * ay * NdotH2 * NdotH2;
        return numbers::inv_pi<scalar_type> * nom / denom;
    }

    scalar_type DG1(scalar_type ndf, scalar_type maxNdotV, scalar_type lambda_V)
    {
        Beckmann<T,false> beckmann;
        scalar_type dg = beckmann.DG1(ndf, maxNdotV, lambda_V);
        onePlusLambda_V = beckmann.onePlusLambda_V;
        return dg;
    }

    scalar_type DG1(scalar_type ndf, scalar_type absNdotV, scalar_type lambda_V, bool transmitted, scalar_type VdotH, scalar_type LdotH, scalar_type VdotHLdotH, scalar_type orientedEta, scalar_type reflectance)
    {
        Beckmann<T,false> beckmann;
        scalar_type dg = beckmann.DG1(ndf, absNdotV, lambda_V, transmitted, VdotH, LdotH, VdotHLdotH, orientedEta, reflectance);
        onePlusLambda_V = beckmann.onePlusLambda_V;
        return dg;
    }

    scalar_type G1(scalar_type lambda)
    {
        return 1.0 / (1.0 + lambda);
    }

    scalar_type C2(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return NdotX2 / (TdotX2 * ax2 + BdotX2 * ay2);
    }

    scalar_type Lambda(scalar_type c2)
    {
        scalar_type c = sqrt<scalar_type>(c2);
        scalar_type nom = 1.0 - 1.259 * c + 0.396 * c2;
        scalar_type denom = 2.181 * c2 + 3.535 * c;
        return hlsl::mix<scalar_type>(0.0, nom / denom, c < 1.6);
    }

    scalar_type Lambda(scalar_type TdotX2, scalar_type BdotX2, scalar_type NdotX2, scalar_type ax2, scalar_type ay2)
    {
        return Lambda(C2(TdotX2, BdotX2, NdotX2, ax2, ay2));
    }

    scalar_type correlated(scalar_type ax2, scalar_type ay2, scalar_type TdotV2, scalar_type BdotV2, scalar_type NdotV2, scalar_type TdotL2, scalar_type BdotL2, scalar_type NdotL2)
    {
        scalar_type c2 = C2(TdotV2, BdotV2, NdotV2, ax2, ay2);
        scalar_type L_v = Lambda(c2);
        c2 = C2(TdotL2, BdotL2, NdotL2, ax2, ay2);
        scalar_type L_l = Lambda(c2);
        return G1(L_v + L_l);
    }

    scalar_type G2_over_G1(scalar_type ax2, scalar_type ay2, scalar_type TdotL2, scalar_type BdotL2, scalar_type NdotL2, scalar_type lambdaV_plus_one)
    {
        scalar_type c2 = C2(TdotL2, BdotL2, NdotL2, ax2, ay2);
        scalar_type lambdaL = Lambda(c2);
        return lambdaV_plus_one / (lambdaV_plus_one + lambdaL);
    }

    scalar_type onePlusLambda_V;
};

}
}
}
}

#endif
