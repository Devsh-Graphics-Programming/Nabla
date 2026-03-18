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

    // InvertibleSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector3_type;
    using density_type = scalar_type;
    using weight_type = density_type;

    struct cache_type
    {
        density_type pdf;
    };

    // NOTE: produces a degenerate (all-zero) bilinear patch when the receiver normal faces away
    // from all three triangle vertices, resulting in NaN PDFs (0 * inf). Callers must ensure
    // at least one vertex has positive projection onto the receiver normal.
    Bilinear<scalar_type> computeBilinearPatch()
    {
        const scalar_type minimumProjSolidAngle = 0.0;

        matrix<T, 3, 3> m = matrix<T, 3, 3>(sphtri.tri_vertices[0], sphtri.tri_vertices[1], sphtri.tri_vertices[2]);
        const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(receiverWasBSDF, hlsl::mul(m, receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));

        return Bilinear<scalar_type>::create(bxdfPdfAtVertex.yyxz);
    }

    codomain_type generate(const domain_type u, NBL_REF_ARG(cache_type) cache)
    {
        Bilinear<scalar_type> bilinear = computeBilinearPatch();
        typename Bilinear<scalar_type>::cache_type bilinearCache;
        const vector2_type warped = bilinear.generate(u, bilinearCache);
        typename SphericalTriangle<scalar_type>::cache_type sphtriCache;
        const vector3_type L = sphtri.generate(warped, sphtriCache);
        // combined weight: sphtri pdf (1/solidAngle) * bilinear pdf at u
        cache.pdf = sphtri.forwardPdf(sphtriCache) * bilinear.forwardPdf(bilinearCache);
        return L;
    }

    density_type forwardPdf(const cache_type cache)
    {
        return cache.pdf;
    }

    weight_type forwardWeight(const cache_type cache)
    {
        return forwardPdf(cache);
    }

    density_type backwardPdf(const vector3_type L)
    {
        const density_type pdf = sphtri.backwardPdf(L);
        const vector2_type u = sphtri.generateInverse(L);
        Bilinear<scalar_type> bilinear = computeBilinearPatch();
        return pdf * bilinear.backwardPdf(u);
    }

    weight_type backwardWeight(const vector3_type L)
    {
        return backwardPdf(L);
    }

    sampling::SphericalTriangle<T> sphtri;
    vector3_type receiverNormal;
    bool receiverWasBSDF;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
