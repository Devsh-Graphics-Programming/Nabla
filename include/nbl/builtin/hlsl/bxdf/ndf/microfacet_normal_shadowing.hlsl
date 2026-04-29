// Copyright (C) 2018-2026 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_BXDF_NDF_MICROFACET_NORMAL_SHADOWING_INCLUDED_
#define _NBL_BUILTIN_HLSL_BXDF_NDF_MICROFACET_NORMAL_SHADOWING_INCLUDED_

namespace nbl
{
namespace hlsl
{
namespace bxdf
{
namespace ndf
{

enum PerturbedNormalShadowing : uint16_t
{
    PNS_SCHUSSLER,
    PNS_YINING
};

template<typename T, PerturbedNormalShadowing P>
struct ShadowingMethod;

// based on Microfacet-based Normal Mapping for Robust Monte Carlo Path Tracing: https://jo.dreggn.org/home/2017_normalmap.pdf
template<typename T>
struct ShadowingMethod<T, PNS_SCHUSSLER>
{
    using scalar_type = T;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    static scalar_type G1(const scalar_type clampedNdotL, const scalar_type NdotNp, const scalar_type clampedNpdotL, const scalar_type clampedNtdotL)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0));
        return hlsl::min(scalar_type(1.0),
            clampedNdotL * hlsl::max(scalar_type(0.0), NdotNp)
            / (clampedNpdotL + clampedNtdotL * sinThetaNp)
        );
    }

    static scalar_type lambdaP(const scalar_type NdotNp, const scalar_type clampedNpdotV, const scalar_type clampedNtdotV)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0));
        return clampedNpdotV / (clampedNpdotV + clampedNtdotV * sinThetaNp);
    }

    static vector3_type computeNt(const vector3_type Np, const matrix3x3_type shadingBasis)
    {
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);
        const vector3_type local_Nt = hlsl::normalize(-vector3_type(local_Np.xy, 0.0));
        return hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
    }
};

// based on Taming the Shadow Terminator: https://www.yiningkarlli.com/projects/shadowterminator/shadow_terminator_v1_1.pdf
template<typename T>
struct ShadowingMethod<T, PNS_YINING>
{
    using scalar_type = T;
    using vector3_type = vector<scalar_type, 3>;
    using matrix3x3_type = matrix<scalar_type, 3, 3>;

    static scalar_type G1(const scalar_type clampedNdotL, const scalar_type NdotNp, const scalar_type clampedNpdotL, const scalar_type clampedNtdotL)
    {
        const scalar_type g = hlsl::min(scalar_type(1.0),
            clampedNdotL / (clampedNpdotL * NdotNp)
        );
        const scalar_type g2 = g * g;
        return -g2 * g + g2 + g;
    }

    // TODO: verify maths
    // since Nt is now perpendicular to Np and not N
    // total area of surface = 1 = hypotenuse of right triangle
    // area of perturbed facet = cos(Np) = NdotNp
    // area of tangent facet = cos(Nt) = sin(Np)
    // projected area of Np onto V = area * NpdotV
    static scalar_type lambdaP(const scalar_type NdotNp, const scalar_type clampedNpdotV, const scalar_type clampedNtdotV)
    {
        const scalar_type sinThetaNp = hlsl::sqrt(hlsl::max(1.0 - NdotNp * NdotNp, 0.0));
        const scalar_type ap = clampedNpdotV * NdotNp;
        if (ap < numeric_limits<scalar_type>::min)
            return scalar_type(0.0);
        const scalar_type at = clampedNtdotV * sinThetaNp;
        return ap / (ap + at);
    }

    static vector3_type computeNt(const vector3_type Np, const matrix3x3_type shadingBasis)
    {
        const vector3_type local_Np = hlsl::mul(shadingBasis, Np);
        const vector3_type local_Nt = hlsl::normalize(vector3_type(local_Np.xy * -local_Np.z, 1.0 - local_Np.z*local_Np.z));
        return hlsl::mul(hlsl::transpose(shadingBasis), local_Nt);
    }
};

}
}
}
}

#endif
