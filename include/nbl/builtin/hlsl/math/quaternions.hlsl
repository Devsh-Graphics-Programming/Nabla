// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
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

    using AsUint = typename unsigned_integer_of_size<sizeof(scalar_type)>::type;

    static this_t create()
    {
        this_t q;
        q.data = data_type(0.0, 0.0, 0.0, 1.0);
        return q;
    }

    // angle: Rotation angle expressed in radians.
    // axis: Rotation axis, must be normalized.
    static this_t create(const vector3_type axis, scalar_type angle)
    {
        this_t q;
        const scalar_type sinTheta = hlsl::sin(angle * 0.5);
        const scalar_type cosTheta = hlsl::cos(angle * 0.5);
        q.data = data_type(axis * sinTheta, cosTheta);
        return q;
    }

    template<typename U=vector<scalar_type,2> NBL_FUNC_REQUIRES(is_same_v<vector<scalar_type,2>,U>)
    static this_t create(const U halfPitchCosSin, const U halfYawCosSin, const U halfRollCosSin)
    {
        const scalar_type cp = halfPitchCosSin.x;
        const scalar_type sp = halfPitchCosSin.y;

        const scalar_type cy = halfYawCosSin.x;
        const scalar_type sy = halfYawCosSin.y;

        const scalar_type cr = halfRollCosSin.x;
        const scalar_type sr = halfRollCosSin.y;

        this_t q;
        q.data[0] = cr * sp * cy + sr * cp * sy; // x
        q.data[1] = cr * cp * sy - sr * sp * cy; // y
        q.data[2] = sr * cp * cy - cr * sp * sy; // z
        q.data[3] = cr * cp * cy + sr * sp * sy; // w

        return q;
    }

    template<typename U=scalar_type NBL_FUNC_REQUIRES(is_same_v<scalar_type,U>)
    static this_t create(const U pitch, const U yaw, const U roll)
    {
        const scalar_type halfPitch = pitch * scalar_type(0.5);
        const scalar_type halfYaw = yaw * scalar_type(0.5);
        const scalar_type halfRoll = roll * scalar_type(0.5);

        return create(
            vector<scalar_type,2>(hlsl::cos(halfPitch), hlsl::sin(halfPitch)),
            vector<scalar_type,2>(hlsl::cos(halfYaw), hlsl::sin(halfYaw)),
            vector<scalar_type,2>(hlsl::cos(halfRoll), hlsl::sin(halfRoll))
        );
    }

    static bool __isEqual(const scalar_type a, const scalar_type b)
    {
        return hlsl::max(a/b, b/a) <= scalar_type(1e-4);
    }
    static bool __dotIsZero(const vector3_type a, const vector3_type b)
    {
        const scalar_type ab = hlsl::dot(a, b);
        return hlsl::abs(ab) <= scalar_type(1e-4);
    }

    static this_t create(NBL_CONST_REF_ARG(matrix_type) m, const bool dontAssertValidMatrix=false)
    {
        {
            // only orthogonal and uniform scale mats can be converted
            bool valid = __dotIsZero(m[0], m[1]);
            valid = __dotIsZero(m[1], m[2]) && valid;
            valid = __dotIsZero(m[0], m[2]) && valid;

            const matrix_type m_T = hlsl::transpose(m);
            const scalar_type dotCol0 = hlsl::dot(m_T[0],m_T[0]);
            const scalar_type dotCol1 = hlsl::dot(m_T[1],m_T[1]);
            const scalar_type dotCol2 = hlsl::dot(m_T[2],m_T[2]);
            valid = __isEqual(dotCol0, dotCol1) && valid;
            valid = __isEqual(dotCol1, dotCol2) && valid;
            valid = __isEqual(dotCol0, dotCol2) && valid;

            if (dontAssertValidMatrix)
                if (!valid)
                {
                    this_t retval;
                    retval.data = hlsl::promote<data_type>(bit_cast<scalar_type>(numeric_limits<scalar_type>::quiet_NaN));
                    return retval;
                }
            else
                assert(valid);
        }

        const scalar_type m00 = m[0][0], m11 = m[1][1], m22 = m[2][2];
        const scalar_type neg_m00 = bit_cast<scalar_type>(bit_cast<AsUint>(m00)^0x80000000u);
        const scalar_type neg_m11 = bit_cast<scalar_type>(bit_cast<AsUint>(m11)^0x80000000u);
        const scalar_type neg_m22 = bit_cast<scalar_type>(bit_cast<AsUint>(m22)^0x80000000u);
        const data_type Qx = data_type(m00, m00, neg_m00, neg_m00);
        const data_type Qy = data_type(m11, neg_m11, m11, neg_m11);
        const data_type Qz = data_type(m22, neg_m22, neg_m22, m22);

        const data_type tmp = hlsl::promote<data_type>(1.0) + Qx + Qy + Qz;

        // TODO: speed this up
        this_t retval;
        if (tmp.x > scalar_type(0.0))
        {
            const scalar_type invscales = scalar_type(0.5) / hlsl::sqrt(tmp.x);
            retval.data.x = (m[2][1] - m[1][2]) * invscales;
            retval.data.y = (m[0][2] - m[2][0]) * invscales;
            retval.data.z = (m[1][0] - m[0][1]) * invscales;
            retval.data.w = tmp.x * invscales * scalar_type(0.5);
        }
        else
        {
            if (tmp.y > scalar_type(0.0))
            {
                const scalar_type invscales = scalar_type(0.5) / hlsl::sqrt(tmp.y);
                retval.data.x = tmp.y * invscales * scalar_type(0.5);
                retval.data.y = (m[0][1] + m[1][0]) * invscales;
                retval.data.z = (m[2][0] + m[0][2]) * invscales;
                retval.data.w = (m[2][1] - m[1][2]) * invscales;
            }
            else if (tmp.z > scalar_type(0.0))
            {
                const scalar_type invscales = scalar_type(0.5) / hlsl::sqrt(tmp.z);
                retval.data.x = (m[0][1] + m[1][0]) * invscales;
                retval.data.y = tmp.z * invscales * scalar_type(0.5);
                retval.data.z = (m[0][2] - m[2][0]) * invscales;
                retval.data.w = (m[1][2] + m[2][1]) * invscales;
            }
            else
            {
                const scalar_type invscales = scalar_type(0.5) / hlsl::sqrt(tmp.w);
                retval.data.x = (m[0][2] + m[2][0]) * invscales;
                retval.data.y = (m[1][2] + m[2][1]) * invscales;
                retval.data.z = tmp.w * invscales * scalar_type(0.5);
                retval.data.w = (m[1][0] - m[0][1]) * invscales;
            }
        }

        retval.data = hlsl::normalize(retval.data);
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
        this_t retval;
        retval.data = data_type(
            data.w * other.data.w - data.x * other.x - data.y * other.data.y - data.z * other.data.z,
            data.w * other.data.x + data.x * other.w + data.y * other.data.z - data.z * other.data.y,
            data.w * other.data.y - data.x * other.z + data.y * other.data.w + data.z * other.data.x,
            data.w * other.data.z + data.x * other.y - data.y * other.data.x + data.z * other.data.w
        );
        return retval;
    }

    static this_t unnormLerp(const this_t start, const this_t end, const scalar_type fraction, const scalar_type totalPseudoAngle)
    {
        // TODO: benchmark uint sign flip vs just *sign(totalPseudoAngle)
        const data_type adjEnd = ieee754::flipSignIfRHSNegative<data_type>(end.data, totalPseudoAngle);

        this_t retval;
        retval.data = hlsl::mix(start.data, adjEnd, fraction);
        return retval;
    }

    static this_t unnormLerp(const this_t start, const this_t end, const scalar_type fraction)
    {
        return unnormLerp(start, end, fraction, hlsl::dot(start.data, end.data));
    }

    static this_t lerp(const this_t start, const this_t end, const scalar_type fraction)
    {
        this_t retval = unnormLerp(start, end, fraction);
        retval.data = hlsl::normalize(retval.data);
        return retval;
    }

    static scalar_type __adj_interpolant(const scalar_type angle, const scalar_type fraction, const scalar_type interpolantPrecalcTerm2, const scalar_type interpolantPrecalcTerm3)
    {
        const scalar_type A = scalar_type(1.0904) + angle * (scalar_type(-3.2452) + angle * (scalar_type(3.55645) - angle * scalar_type(1.43519)));
        const scalar_type B = scalar_type(0.848013) + angle * (scalar_type(-1.06021) + angle * scalar_type(0.215638));
        const scalar_type k = A * interpolantPrecalcTerm2 + B;
        return fraction + interpolantPrecalcTerm3 * k;
    }

    static this_t unnormFlerp(const this_t start, const this_t end, const scalar_type fraction)
    {
        const scalar_type pseudoAngle = hlsl::dot(start.data,end.data);
        const scalar_type interpolantPrecalcTerm = fraction - scalar_type(0.5);
        const scalar_type interpolantPrecalcTerm3 = fraction * interpolantPrecalcTerm * (fraction - scalar_type(1.0));
        const scalar_type adjFrac = __adj_interpolant(hlsl::abs(pseudoAngle),fraction,interpolantPrecalcTerm*interpolantPrecalcTerm,interpolantPrecalcTerm3);
        
        this_t retval = unnormLerp(start,end,adjFrac,pseudoAngle);
        return retval;
    }

    static this_t flerp(const this_t start, const this_t end, const scalar_type fraction)
    {       
        this_t retval = unnormFlerp(start,end,fraction);
        retval.data = hlsl::normalize(retval.data);
        return retval;
    }

    vector3_type transformVector(const vector3_type v, const bool assumeNoScale=false) NBL_CONST_MEMBER_FUNC
    {
        scalar_type scale = hlsl::mix(hlsl::length(data), scalar_type(1.0), assumeNoScale);
        vector3_type direction = data.xyz;
        return v * scale + hlsl::cross(direction, v * data.w + hlsl::cross(direction, v)) * scalar_type(2.0);
    }

    matrix_type constructMatrix() NBL_CONST_MEMBER_FUNC
    {
        matrix_type mat;
        mat[0] = data.yzx * data.ywz + data.zxy * data.zyw * vector3_type( 1.0, 1.0,-1.0);
        mat[1] = data.yzx * data.xzw + data.zxy * data.wxz * vector3_type(-1.0, 1.0, 1.0);
        mat[2] = data.yzx * data.wyx + data.zxy * data.xwy * vector3_type( 1.0,-1.0, 1.0);
        mat[0][0] = scalar_type(0.5) - mat[0][0];
        mat[1][1] = scalar_type(0.5) - mat[1][1];
        mat[2][2] = scalar_type(0.5) - mat[2][2];
        mat[0] = mat[0] * scalar_type(2.0);
        mat[1] = mat[1] * scalar_type(2.0);
        mat[2] = mat[2] * scalar_type(2.0);
        return mat;// hlsl::transpose(mat);    // TODO: double check transpose?
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

    this_t inverse() NBL_CONST_MEMBER_FUNC
    {
        this_t retval;
        retval.data.xyz = -retval.data.xyz;
        retval.data.w = data.w;
        return retval;
    }

    data_type data;
};

}


namespace cpp_compat_intrinsics_impl
{
template<typename T>
struct normalize_helper<math::truncated_quaternion<T> >
{
    static inline math::truncated_quaternion<T> __call(const math::truncated_quaternion<T> q)
    {
        math::truncated_quaternion<T> retval;
        retval.data = hlsl::normalize(q.data);
        return retval;
    }
};

template<typename T>
struct normalize_helper<math::quaternion<T> >
{
    static inline math::quaternion<T> __call(const math::quaternion<T> q)
    {
        math::quaternion<T> retval;
        retval.data = hlsl::normalize(q.data);
        return retval;
    }
};
}

namespace impl
{
template<typename T>
struct static_cast_helper<math::quaternion<T>, math::truncated_quaternion<T> >
{
    static inline math::quaternion<T> cast(const math::truncated_quaternion<T> q)
    {
        math::quaternion<T> retval;
        retval.data.xyz = q.data;
        retval.data.w = hlsl::sqrt(scalar_type(1.0) - hlsl::dot(q.data, q.data));
        return retval;
    }
};

template<typename T>
struct static_cast_helper<math::truncated_quaternion<T>, math::quaternion<T> >
{
    static inline math::truncated_quaternion<T> cast(const math::quaternion<T> q)
    {
        math::truncated_quaternion<T> t;
        t.data.x = t.data.x;
        t.data.y = t.data.y;
        t.data.z = t.data.z;
        return t;
    }
};

template<typename T>
struct static_cast_helper<matrix<T,3,3>, math::quaternion<T> >
{
    static inline matrix<T,3,3> cast(const math::quaternion<T> q)
    {
        return q.constructMatrix();
    }
};
}

}
}

#endif
