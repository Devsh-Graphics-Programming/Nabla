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

template<typename T>
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
        typename Bilinear<scalar_type>::cache_type bilinearCache;
    };

    // NOTE: produces a degenerate (all-zero) bilinear patch when the receiver normal faces away
    // from all three triangle vertices, resulting in NaN PDFs (0 * inf). Callers must ensure
    // at least one vertex has positive projection onto the receiver normal.
    static ProjectedSphericalTriangle<T> create(NBL_REF_ARG(shapes::SphericalTriangle<T>) shape, const vector3_type receiverNormal, const bool receiverWasBSDF)
    {
        ProjectedSphericalTriangle<T> retval;
        retval.sphtri = SphericalTriangle<T>::create(shape);

        const scalar_type minimumProjSolidAngle = 0.0;
        matrix<T, 3, 3> m = matrix<T, 3, 3>(shape.vertices[0], shape.vertices[1], shape.vertices[2]);
        const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(receiverWasBSDF, hlsl::mul(m, receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));
        retval.bilinearPatch = Bilinear<scalar_type>::create(bxdfPdfAtVertex.yyxz);

        const scalar_type projSA = shape.projectedSolidAngle(receiverNormal);
        retval.rcpProjSolidAngle = projSA > scalar_type(0.0) ? scalar_type(1.0) / projSA : scalar_type(0.0);
        return retval;
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
    {
        Bilinear<scalar_type> bilinear = bilinearPatch;
        const vector2_type warped = bilinear.generate(u, cache.bilinearCache);
        typename SphericalTriangle<scalar_type>::cache_type sphtriCache;
        const vector3_type L = sphtri.generate(warped, sphtriCache);
        cache.abs_cos_theta = bilinear.forwardWeight(cache.bilinearCache);
        return L;
    }

    density_type forwardPdf(const cache_type cache)
    {
        return sphtri.rcpSolidAngle * bilinearPatch.forwardPdf(cache.bilinearCache);
    }

    weight_type forwardWeight(const cache_type cache)
    {
        return cache.abs_cos_theta * rcpProjSolidAngle;
    }

    density_type backwardPdf(const vector3_type L)
    {
        const vector2_type u = sphtri.generateInverse(L);
        return sphtri.rcpSolidAngle * bilinearPatch.backwardPdf(u);
    }

    weight_type backwardWeight(const vector3_type L)
    {
        cache_type cache;
        // cache.bilinearCache; // unused
        // approximate abs(cos_theta) via bilinear interpolation at the inverse-mapped point
        const vector2_type u = sphtri.generateInverse(L);
        cache.abs_cos_theta = bilinearPatch.backwardWeight(u);
        return forwardWeight(cache);
    }

    sampling::SphericalTriangle<T> sphtri;
    Bilinear<scalar_type> bilinearPatch;
    scalar_type rcpProjSolidAngle;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
