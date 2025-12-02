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

template<typename T>
struct truncated_quaternion
{
    using this_t = truncated_quaternion<T>;
    using scalar_type = T;
    using data_type = vector<T, 3>;

    static this_t create()
    {
        this_t q;
        q.data = data_type(0.0, 0.0, 0.0);
        return q;
    }

    data_type data;
};

template <typename T>
struct quaternion
{
    using this_t = quaternion<T>;
    using scalar_type = T;
    using data_type = vector<T, 4>;
    using vector3_type = vector<T, 3>;
    using matrix_type = matrix<T, 3, 3>;

    static this_t create()
    {
        this_t q;
        q.data = data_type(0.0, 0.0, 0.0, 1.0);
        return q;
    }
    
    static this_t create(scalar_type x, scalar_type y, scalar_type z, scalar_type w)
    {
        this_t q;
        q.data = data_type(x, y, z, w);
        return q;
    }

    static this_t create(NBL_CONST_REF_ARG(this_t) other)
    {
        return other;
    }

    // angle: Rotation angle expressed in radians.
    // axis: Rotation axis, must be normalized.
    static this_t create(scalar_type angle, const vector3_type axis)
    {
        this_t q;
        const scalar_type sinTheta = hlsl::sin(angle * 0.5);
        const scalar_type cosTheta = hlsl::cos(angle * 0.5);
        q.data = data_type(axis.x * sinTheta, axis.y * sinTheta, axis.z * sinTheta, cosTheta);
        return q;
    }


    static this_t create(scalar_type pitch, scalar_type yaw, scalar_type roll)
    {
        const scalar_type rollDiv2 = roll * scalar_type(0.5);
        const scalar_type sr = hlsl::sin(rollDiv2);
        const scalar_type cr = hlsl::cos(rollDiv2);

        const scalar_type pitchDiv2 = pitch * scalar_type(0.5);
        const scalar_type sp = hlsl::sin(pitchDiv2);
        const scalar_type cp = hlsl::cos(pitchDiv2);

        const scalar_type yawDiv2 = yaw * scalar_type(0.5);
        const scalar_type sy = hlsl::sin(yawDiv2);
        const scalar_type cy = hlsl::cos(yawDiv2);

        this_t output;
        output.data[0] = cr * sp * cy + sr * cp * sy; // x
        output.data[1] = cr * cp * sy - sr * sp * cy; // y
        output.data[2] = sr * cp * cy - cr * sp * sy; // z
        output.data[3] = cr * cp * cy + sr * sp * sy; // w

        return output;
    }

    static this_t create(NBL_CONST_REF_ARG(matrix_type) m)
    {
        using AsUint = typename unsigned_integer_of_size<sizeof(scalar_type)>::type;
        const scalar_type m00 = m[0][0], m11 = m[1][1], m22 = m[2][2];
        const scalar_type neg_m00 = bit_cast<scalar_type>(bit_cast<AsUint>(m00)^0x80000000u);
        const scalar_type neg_m11 = bit_cast<scalar_type>(bit_cast<AsUint>(m11)^0x80000000u);
        const scalar_type neg_m22 = bit_cast<scalar_type>(bit_cast<AsUint>(m22)^0x80000000u);
        const data_type Qx = data_type(m00, m00, neg_m00, neg_m00);
        const data_type Qy = data_type(m11, neg_m11, m11, neg_m11);
        const data_type Qz = data_type(m22, neg_m22, neg_m22, m22);

        const data_type tmp = hlsl::promote<data_type>(1.0) + Qx + Qy + Qz;
        const data_type invscales = hlsl::promote<data_type>(0.5) / hlsl::sqrt(tmp);
        const data_type scales = tmp * invscales * hlsl::promote<data_type>(0.5);

        // TODO: speed this up
        this_t retval;
        if (tmp.x > scalar_type(0.0))
        {
            retval.data.x = (m[2][1] - m[1][2]) * invscales.x;
            retval.data.y = (m[0][2] - m[2][0]) * invscales.x;
            retval.data.z = (m[1][0] - m[0][1]) * invscales.x;
            retval.data.w = scales.x;
        }
        else
        {
            if (tmp.y > scalar_type(0.0))
            {
                retval.data.x = scales.y;
                retval.data.y = (m[0][1] + m[1][0]) * invscales.y;
                retval.data.z = (m[2][0] + m[0][2]) * invscales.y;
                retval.data.w = (m[2][1] - m[1][2]) * invscales.y;
            }
            else if (tmp.z > scalar_type(0.0))
            {
                retval.data.x = (m[0][1] + m[1][0]) * invscales.z;
                retval.data.y = scales.z;
                retval.data.z = (m[0][2] - m[2][0]) * invscales.z;
                retval.data.w = (m[1][2] + m[2][1]) * invscales.z;
            }
            else
            {
                retval.data.x = (m[0][2] + m[2][0]) * invscales.w;
                retval.data.y = (m[1][2] + m[2][1]) * invscales.w;
                retval.data.z = scales.w;
                retval.data.w = (m[1][0] - m[0][1]) * invscales.w;
            }
        }

        retval.data = hlsl::normalize(retval.data);
        return retval;
    }

    static this_t createFromTruncated(NBL_CONST_REF_ARG(truncated_quaternion<T>) first3Components)
    {
        this_t retval;
        retval.data.xyz = first3Components.data;
        retval.data.w = hlsl::sqrt(scalar_type(1.0) - hlsl::dot(first3Components.data, first3Components.data));
        return retval;
    }

    this_t operator*(scalar_type scalar)
    {
        this_t output;
        output.data = data * scalar;
        return output;
    }

    this_t operator*(NBL_CONST_REF_ARG(this_t) other)
    {
        return this_t::create(
            data.w * other.data.w - data.x * other.x - data.y * other.data.y - data.z * other.data.z,
            data.w * other.data.x + data.x * other.w + data.y * other.data.z - data.z * other.data.y,
            data.w * other.data.y - data.x * other.z + data.y * other.data.w + data.z * other.data.x,
            data.w * other.data.z + data.x * other.y - data.y * other.data.x + data.z * other.data.w
        );
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

namespace impl
{
template<typename T>
struct static_cast_helper<matrix<T,3,3>, math::quaternion<T> >
{
    static inline matrix<T,3,3> cast(math::quaternion<T> q)
    {
        return q.constructMatrix();
    }
};
}

}
}

#endif
