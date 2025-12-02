// Copyright (C) 2018-2023 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_QUATERNIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_QUATERNIONS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"

namespace nbl
{
namespace hlsl
{
namespace math
{

template <typename T>
struct quaternion
{
    using this_t = quaternion<T>;
    using scalar_type = T;
    using data_type = vector<T, 4>;
    using vector3_type = vector<T, 3>;
    using matrix_type = matrix<T, 3, 3>;

    static this_t createFromTruncated(const vector3_type first3Components)
    {
        this_t retval;
        retval.data.xyz = first3Components;
        retval.data.w = hlsl::sqrt(scalar_type(1.0) - hlsl::dot(first3Components, first3Components));
        return retval;
    }

    static this_t lerp(const this_t start, const this_t end, const scalar_type fraction, const scalar_type totalPseudoAngle)
    {
        using AsUint = typename unsigned_integer_of_size<sizeof(scalar_type)>::type;
        const AsUint negationMask = hlsl::bit_cast<AsUint>(totalPseudoAngle) & AsUint(0x80000000u);
        const data_type adjEnd = hlsl::bit_cast<scalar_type>(hlsl::bit_cast<AsUint>(end.data) ^ negationMask);

        this_t retval;
        retval.data = hlsl::mix(start.data, adjEnd, fraction);
        return retval;
    }

    static this_t lerp(const this_t start, const this_t end, const scalar_type fraction)
    {
        return lerp(start, end, fraction, hlsl::dot(start.data, end.data));
    }

    static scalar_type __adj_interpolant(const scalar_type angle, const scalar_type fraction, const scalar_type interpolantPrecalcTerm2, const scalar_type interpolantPrecalcTerm3)
    {
        const scalar_type A = scalar_type(1.0904) + angle * (scalar_type(-3.2452) + angle * (scalar_type(3.55645) - angle * scalar_type(1.43519)));
        const scalar_type B = scalar_type(0.848013) + angle * (scalar_type(-1.06021) + angle * scalar_type(0.215638));
        const scalar_type k = A * interpolantPrecalcTerm2 + B;
        return fraction + interpolantPrecalcTerm3 * k;
    }

    static this_t flerp(const this_t start, const this_t end, const scalar_type fraction)
    {
        const scalar_type pseudoAngle = hlsl::dot(start.data,end.data);
        const scalar_type interpolantPrecalcTerm = fraction - scalar_type(0.5);
        const scalar_type interpolantPrecalcTerm3 = fraction * interpolantPrecalcTerm * (fraction - scalar_type(1.0));
        const scalar_type adjFrac = __adj_interpolant(hlsl::abs(pseudoAngle),fraction,interpolantPrecalcTerm*interpolantPrecalcTerm,interpolantPrecalcTerm3);
        
        this_t retval = lerp(start,end,adjFrac,pseudoAngle);
        retval.data = hlsl::normalize(retval.data);
        return retval;
    }

    matrix_type constructMatrix()
    {
        matrix_type mat;
        mat[0] = data.yzx * data.ywz + data.zxy * data.zyw * vector3_type( 1.0, 1.0,-1.0);
        mat[1] = data.yzx * data.xzw + data.zxy * data.wxz * vector3_type(-1.0, 1.0, 1.0);
        mat[2] = data.yzx * data.wyx + data.zxy * data.xwy * vector3_type( 1.0,-1.0, 1.0);
        mat[0][0] = scalar_type(0.5) - mat[0][0];
        mat[1][1] = scalar_type(0.5) - mat[1][1];
        mat[2][2] = scalar_type(0.5) - mat[2][2];
        mat *= scalar_type(2.0);
        return hlsl::transpose(mat);    // TODO: double check transpose?
    }

    static vector3_type slerp_delta(const vector3_type start, const vector3_type preScaledWaypoint, scalar_type cosAngleFromStart)
    {
        vector3_type planeNormal = hlsl::cross(start,preScaledWaypoint);
    
        cosAngleFromStart *= scalar_type(0.5);
        const scalar_type sinAngle = hlsl::sqrt(scalar_type(0.5) - cosAngleFromStart);
        const scalar_type cosAngle = hlsl::sqrt(scalar_type(0.5) + cosAngleFromStart);
        
        planeNormal *= sinAngle;
        const vector3_type precompPart = hlsl::cross(planeNormal, start) * scalar_type(2.0);

        return precompPart * cosAngle + hlsl::cross(planeNormal, precompPart);
    }

    data_type data;
};

}
}
}

#endif
