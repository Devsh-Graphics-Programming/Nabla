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
        scalar_type abs_cos_theta;
        vector2_type warped;
        typename Bilinear<scalar_type>::cache_type bilinearCache;
    };

    // NOTE: produces a degenerate (all-zero) bilinear patch when the receiver normal faces away
    // from all three triangle vertices, resulting in NaN PDFs (0 * inf). Callers must ensure
    // at least one vertex has positive projection onto the receiver normal.
    static ProjectedSphericalTriangle<T,UsePdfAsWeight> create(NBL_REF_ARG(shapes::SphericalTriangle<T>) shape, const vector3_type _receiverNormal, const bool _receiverWasBSDF)
    {
        ProjectedSphericalTriangle<T,UsePdfAsWeight> retval;
       retval.sphtri = SphericalTriangle<T, UsePdfAsWeight>::create(shape);
        retval.receiverNormal = _receiverNormal;
        retval.receiverWasBSDF = _receiverWasBSDF;

        const scalar_type minimumProjSolidAngle = 0.0;
        matrix<T, 3, 3> m = matrix<T, 3, 3>(shape.vertices[0], shape.vertices[1], shape.vertices[2]);
        const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(_receiverWasBSDF, hlsl::mul(m, _receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));
        retval.bilinearPatch = Bilinear<scalar_type>::create(bxdfPdfAtVertex.yyxz);

        const scalar_type projSA = shape.projectedSolidAngle(_receiverNormal);
        retval.rcpProjSolidAngle = projSA > scalar_type(0.0) ? scalar_type(1.0) / projSA : scalar_type(0.0);
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache) NBL_CONST_MEMBER_FUNC
    {
        Bilinear<scalar_type> bilinear = bilinearPatch;
        cache.warped = bilinear.generate(u, cache.bilinearCache);
        typename SphericalTriangle<scalar_type>::cache_type sphtriCache;
        const vector3_type L = sphtri.generate(cache.warped, sphtriCache);
        cache.abs_cos_theta = bilinear.forwardWeight(u, cache.bilinearCache);
        return L;
    }

    density_type forwardPdf(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        return sphtri.rcpSolidAngle * bilinearPatch.forwardPdf(u, cache.bilinearCache);
    }

    weight_type forwardWeight(const domain_type u, const cache_type cache) NBL_CONST_MEMBER_FUNC
    {
        if (UsePdfAsWeight)
            return forwardPdf(u, cache);
        return cache.abs_cos_theta * rcpProjSolidAngle;
    }

    density_type backwardPdf(const codomain_type L) NBL_CONST_MEMBER_FUNC
    {
        const vector2_type u = sphtri.generateInverse(L);
        return sphtri.rcpSolidAngle * bilinearPatch.backwardPdf(u);
    }

    weight_type backwardWeight(const codomain_type L) NBL_CONST_MEMBER_FUNC
    {
        NBL_IF_CONSTEXPR(UsePdfAsWeight)
            return backwardPdf(L);
        const scalar_type minimumProjSolidAngle = 0.0;
        const scalar_type abs_cos_theta = math::conditionalAbsOrMax(receiverWasBSDF, hlsl::dot(receiverNormal, L), minimumProjSolidAngle);
        return abs_cos_theta * rcpProjSolidAngle;
    }

    sampling::SphericalTriangle<T, UsePdfAsWeight> sphtri;
    Bilinear<scalar_type> bilinearPatch;
    scalar_type rcpProjSolidAngle;
    vector3_type receiverNormal;
    bool receiverWasBSDF;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
