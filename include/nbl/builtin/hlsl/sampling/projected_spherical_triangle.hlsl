// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_PROJECTED_SPHERICAL_TRIANGLE_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/math/functions.hlsl>
#include <nbl/builtin/hlsl/sampling/bilinear.hlsl>
#include <nbl/builtin/hlsl/sampling/spherical_triangle.hlsl>

namespace nbl
{
namespace hlsl
{
namespace sampling
{

// "Practical Warps" projected solid angle sampler for spherical triangles.
//
// How it works:
//   1. Build a bilinear patch from NdotL at each vertex direction
//   2. Warp uniform [0,1]^2 through the bilinear to importance-sample NdotL
//   3. Feed the warped UV into the solid angle sampler to get a direction
//   4. PDF = (1/SolidAngle) * bilinearPdf
//
// Template parameter `UsePdfAsWeight`: when true (default), forwardWeight/backwardWeight
// return the PDF instead of the projected-solid-angle MIS weight.
// TODO: the projected-solid-angle MIS weight (UsePdfAsWeight=false) has been shown to be
// poor in practice. Once confirmed by testing, remove the false path and stop storing
// receiverNormal, receiverWasBSDF, and rcpProjSolidAngle as members.
template<typename T, bool UsePdfAsWeight = true>
struct ProjectedSphericalTriangle
{
    using scalar_type = T;
    using vector2_type = vector<T, 2>;
    using vector3_type = vector<T, 3>;
    using vector4_type = vector<T, 4>;

    // BackwardTractableSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector3_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        typename Bilinear<scalar_type>::cache_type bilinearCache;
        vector3_type L; // TODO: erase when UsePdfAsWeight==false
    };

    // Shouldn't produce NAN if all corners have 0 proj solid angle due to min density adds/clamps in the linear sampler
    static ProjectedSphericalTriangle<T,UsePdfAsWeight> create(NBL_REF_ARG(shapes::SphericalTriangle<T>) shape, const vector3_type _receiverNormal, const bool _receiverWasBSDF)
    {
        ProjectedSphericalTriangle<T,UsePdfAsWeight> retval;
        retval.sphtri = SphericalTriangle<T, UsePdfAsWeight>::create(shape);

        const scalar_type minimumProjSolidAngle = 0.0;
        matrix<T, 3, 3> m = matrix<T, 3, 3>(shape.vertices[0], shape.vertices[1], shape.vertices[2]);
        const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(_receiverWasBSDF, hlsl::mul(m, _receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));
        retval.bilinearPatch = Bilinear<scalar_type>::create(bxdfPdfAtVertex.yyxz);

        NBL_IF_CONSTEXPR(!UsePdfAsWeight)
        {
            retval.receiverNormal = _receiverNormal;
            // prevent division of 0 cosine by 0
            retval.projSolidAngle = max<scalar_type>(shape.projectedSolidAngle(_receiverNormal),numeric_limits<scalar_type>::min);
        }
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        const vector2_type warped = bilinearPatch.generate(u, cache.bilinearCache);
        typename SphericalTriangle<scalar_type>::cache_type sphtriCache; // PDF is constant caches nothing, its empty
        const codomain_type L = sphtri.generate(warped, sphtriCache);
        NBL_IF_CONSTEXPR(!UsePdfAsWeight)
            cache.L = L;
        return L;
    }

    density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return sphtri.getRcpSolidAngle() * bilinearPatch.forwardPdf(u,cache.bilinearCache);
    }

    weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        NBL_IF_CONSTEXPR (UsePdfAsWeight)
            return forwardPdf(u,cache);
        return backwardWeight(cache.L);
    }

    weight_type backwardWeight(const codomain_type L) NBL_CONST_MEMBER_FUNC
    {
        NBL_IF_CONSTEXPR (UsePdfAsWeight)
        {
            const vector2_type u = sphtri.generateInverse(L);
            return sphtri.getRcpSolidAngle() * bilinearPatch.backwardPdf(u);
        }
        // make the MIS weight always abs because even when receiver is a BRDF, the samples in lower hemisphere will get killed and MIS weight never used
        return hlsl::abs(hlsl::dot(L,receiverNormal))/projSolidAngle;
    }

    sampling::SphericalTriangle<T, UsePdfAsWeight> sphtri;
    Bilinear<scalar_type> bilinearPatch;
    // TODO: erase when UsePdfAsWeight==false
    vector3_type receiverNormal;
    vector3_type projSolidAngle;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
