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
#include <nbl/builtin/hlsl/sampling/warp_and_pdf.hlsl>

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

    // ResamplableSampler concept types
    using domain_type = vector2_type;
    using codomain_type = vector3_type;
    using density_type = scalar_type;
    using sample_type = codomain_and_rcpPdf<codomain_type, density_type>;

    Bilinear<scalar_type> computeBilinearPatch()
    {
        const scalar_type minimumProjSolidAngle = 0.0;

        matrix<T, 3, 3> m = matrix<T, 3, 3>(sphtri.tri.vertices[0], sphtri.tri.vertices[1], sphtri.tri.vertices[2]);
        const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(receiverWasBSDF, hlsl::mul(m, receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));

        return Bilinear<scalar_type>::create(bxdfPdfAtVertex.yyxz);
    }

    vector3_type generate(const vector2_type u)
    {
        vector2_type u;
        // pre-warp according to proj solid angle approximation
        Bilinear<scalar_type> bilinear = computeBilinearPatch();
        u = bilinear.generate(_u);

        // now warp the points onto a spherical triangle
        const vector3_type L = sphtri.generate(u);
        return L;
    }

    scalar_type forwardPdf(const vector2_type u)
    {
        const scalar_type pdf = sphtri.forwardPdf(u);
        Bilinear<scalar_type> bilinear = computeBilinearPatch();
        return pdf * bilinear.backwardPdf(u);
    }

    scalar_type backwardPdf(const vector3_type L)
    {
        const scalar_type pdf = sphtri.backwardPdf(L);
        const vector2_type u = sphtri.generateInverse(L);
        Bilinear<scalar_type> bilinear = computeBilinearPatch();
        return pdf * bilinear.backwardPdf(u);
    }

    sampling::SphericalTriangle<T> sphtri;
    vector3_type receiverNormal;
    bool receiverWasBSDF;
};

} // namespace sampling
} // namespace hlsl
} // namespace nbl

#endif
