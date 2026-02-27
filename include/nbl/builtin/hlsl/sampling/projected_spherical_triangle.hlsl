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

    static ProjectedSphericalTriangle<T> create(NBL_CONST_REF_ARG(shapes::SphericalTriangle<T>) tri)
    {
        ProjectedSphericalTriangle<T> retval;
        retval.sphtri = sampling::SphericalTriangle<T>::create(tri);
        return retval;
    }

    vector4_type computeBilinearPatch(const vector3_type receiverNormal, bool isBSDF)
    {
        const scalar_type minimumProjSolidAngle = 0.0;

        matrix<T, 3, 3> m = matrix<T, 3, 3>(sphtri.tri.vertices[0], sphtri.tri.vertices[1], sphtri.tri.vertices[2]);
        const vector3_type bxdfPdfAtVertex = math::conditionalAbsOrMax(isBSDF, hlsl::mul(m, receiverNormal), hlsl::promote<vector3_type>(minimumProjSolidAngle));

        return bxdfPdfAtVertex.yyxz;
    }

    vector3_type generate(NBL_REF_ARG(scalar_type) rcpPdf, scalar_type solidAngle, scalar_type cos_c, scalar_type csc_b, const vector3_type receiverNormal, bool isBSDF, const vector2_type _u)
    {
        vector2_type u;
        // pre-warp according to proj solid angle approximation
        vector4_type patch = computeBilinearPatch(receiverNormal, isBSDF);
        Bilinear<scalar_type> bilinear = Bilinear<scalar_type>::create(patch);
        u = bilinear.generate(_u);

        // now warp the points onto a spherical triangle
        const vector3_type L = sphtri.generate(cos_c, csc_b, u);
        rcpPdf = solidAngle / bilinear.backwardPdf(u);

        return L;
    }

    vector3_type generate(NBL_REF_ARG(scalar_type) rcpPdf, const vector3_type receiverNormal, bool isBSDF, const vector2_type u)
    {
        const scalar_type cos_c = sphtri.tri.cos_sides[2];
        const scalar_type csc_b = sphtri.tri.csc_sides[1];
        const scalar_type solidAngle = sphtri.tri.solidAngle();
        return generate(rcpPdf, solidAngle, cos_c, csc_b, receiverNormal, isBSDF, u);
    }

    scalar_type backwardPdf(const vector3_type receiverNormal, bool receiverWasBSDF, const vector3_type L)
    {
        scalar_type pdf;
        const vector2_type u = sphtri.generateInverse(pdf, L);

        vector4_type patch = computeBilinearPatch(receiverNormal, receiverWasBSDF);
        Bilinear<scalar_type> bilinear = Bilinear<scalar_type>::create(patch);
        return pdf * bilinear.backwardPdf(u);
    }

    sampling::SphericalTriangle<T> sphtri;
};

}
}
}

#endif
