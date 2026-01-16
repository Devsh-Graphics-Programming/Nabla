// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h
#ifndef _NBL_BUILTIN_HLSL_MATH_QUATERNIONS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_QUATERNIONS_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/tgmath.hlsl"
#include "nbl/builtin/hlsl/math/linalg/matrix_runtime_traits.hlsl"

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
    template<typename U=vector3_type NBL_FUNC_REQUIRES(is_same_v<vector3_type,U>)
    static this_t create(const U axis, const typename vector_traits<U>::scalar_type angle, const typename vector_traits<U>::scalar_type uniformScale = typename vector_traits<U>::scalar_type(1.0))
    {
        using scalar_t = typename vector_traits<U>::scalar_type;
        this_t q;
        const scalar_t halfAngle = angle * scalar_t(0.5);
        const scalar_t sinTheta = hlsl::sin(halfAngle);
        const scalar_t cosTheta = hlsl::cos(halfAngle);
        q.data = data_type(axis * sinTheta, cosTheta) * uniformScale;
        return q;
    }

    // applies rotation equivalent to 3x3 matrix in order of pitch * yaw * roll (X * Y * Z) -- mul(roll,mul(yaw,mul(pitch,v)))
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

    static this_t create(NBL_CONST_REF_ARG(matrix_type) _m, const bool dontAssertValidMatrix=false)
    {
        scalar_type uniformScaleSq;
        {
            // only orthogonal and uniform scale mats can be converted
            linalg::RuntimeTraits<matrix_type> traits = linalg::RuntimeTraits<matrix_type>::create(_m);
            bool valid = traits.orthogonal && !hlsl::isnan(traits.uniformScaleSq);
            uniformScaleSq = traits.uniformScaleSq;

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
        if (uniformScaleSq < numeric_limits<scalar_type>::min)
        {
            this_t retval;
            retval.data = hlsl::promote<data_type>(bit_cast<scalar_type>(numeric_limits<scalar_type>::quiet_NaN));
            return retval;
        }

        const scalar_type uniformScale = hlsl::sqrt(uniformScaleSq);
        matrix_type m = _m;
        m /= uniformScale;

        const scalar_type m00 = m[0][0], m11 = m[1][1], m22 = m[2][2];
        const scalar_type neg_m00 = -m00;
        const scalar_type neg_m11 = -m11;
        const scalar_type neg_m22 = -m22;
        const data_type Qx = data_type(m00, m00, neg_m00, neg_m00);
        const data_type Qy = data_type(m11, neg_m11, m11, neg_m11);
        const data_type Qz = data_type(m22, neg_m22, neg_m22, m22);

        const data_type tmp = Qx + Qy + Qz;

        // TODO: speed this up
        this_t retval;
        if (tmp.x > scalar_type(0.0))
        {
            const scalar_type scales = hlsl::sqrt(tmp.x + scalar_type(1.0));
            const scalar_type invscales = scalar_type(0.5) / scales;
            retval.data.x = (m[1][2] - m[2][1]) * invscales;
            retval.data.y = (m[2][0] - m[0][2]) * invscales;
            retval.data.z = (m[0][1] - m[1][0]) * invscales;
            retval.data.w = scales * scalar_type(0.5);
        }
        else
        {
            if (tmp.y > scalar_type(0.0))
            {
                const scalar_type scales = hlsl::sqrt(tmp.y + scalar_type(1.0));
                const scalar_type invscales = scalar_type(0.5) / scales;
                retval.data.x = scales * scalar_type(0.5);
                retval.data.y = (m[1][0] + m[0][1]) * invscales;
                retval.data.z = (m[0][2] + m[2][0]) * invscales;
                retval.data.w = (m[1][2] - m[2][1]) * invscales;
            }
            else if (tmp.z > scalar_type(0.0))
            {
                const scalar_type scales = hlsl::sqrt(tmp.z + scalar_type(1.0));
                const scalar_type invscales = scalar_type(0.5) / scales;
                retval.data.x = (m[1][0] + m[0][1]) * invscales;
                retval.data.y = scales * scalar_type(0.5);
                retval.data.z = (m[2][1] + m[1][2]) * invscales;
                retval.data.w = (m[2][0] - m[0][2]) * invscales;
            }
            else
            {
                const scalar_type scales = hlsl::sqrt(tmp.w + scalar_type(1.0));
                const scalar_type invscales = scalar_type(0.5) / scales;
                retval.data.x = (m[2][0] + m[0][2]) * invscales;
                retval.data.y = (m[2][1] + m[1][2]) * invscales;
                retval.data.z = scales * scalar_type(0.5);
                retval.data.w = (m[0][1] - m[1][0]) * invscales;
            }
        }

        retval.data = retval.data * uniformScale; // restore uniform scale
        return retval;
    }

    this_t operator*(scalar_type scalar) NBL_CONST_MEMBER_FUNC
    {
        this_t output;
        output.data = data * scalar;
        return output;
    }

    this_t operator*(NBL_CONST_REF_ARG(this_t) other) NBL_CONST_MEMBER_FUNC
    {
        this_t retval;
        retval.data = data_type(
            data.w * other.data.x + data.x * other.data.w + data.y * other.data.z - data.z * other.data.y,
            data.w * other.data.y - data.x * other.data.z + data.y * other.data.w + data.z * other.data.x,
            data.w * other.data.z + data.x * other.data.y - data.y * other.data.x + data.z * other.data.w,
            data.w * other.data.w - data.x * other.data.x - data.y * other.data.y - data.z * other.data.z
        );
        return retval;
    }

    static this_t unnormLerp(const this_t start, const this_t end, const scalar_type fraction, const scalar_type totalPseudoAngle)
    {
        assert(testing::relativeApproxCompare(hlsl::length(start.data), scalar_type(1.0), scalar_type(1e-4)));
        assert(testing::relativeApproxCompare(hlsl::length(end.data), scalar_type(1.0), scalar_type(1e-4)));
        const data_type adjEnd = ieee754::flipSignIfRHSNegative<data_type,scalar_type>(end.data, totalPseudoAngle);

        this_t retval;
        retval.data = hlsl::mix(start.data, adjEnd, hlsl::promote<data_type>(fraction));
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
        assert(testing::relativeApproxCompare(hlsl::length(start.data), scalar_type(1.0), scalar_type(1e-4)));
        assert(testing::relativeApproxCompare(hlsl::length(end.data), scalar_type(1.0), scalar_type(1e-4)));

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
        const scalar_type scaleRcp = hlsl::rsqrt(hlsl::dot(data, data));
        vector3_type retV = v;
        scalar_type modVScale = scalar_type(2.0);
        if (!assumeNoScale)
        {
            retV /= scaleRcp;
            modVScale *= scaleRcp;
        }
        const vector3_type modV = v * modVScale;
        const vector3_type direction = data.xyz;
        return retV + hlsl::cross(direction, modV * data.w + hlsl::cross(direction, modV));
    }

    matrix_type __constructMatrix() NBL_CONST_MEMBER_FUNC
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
        return mat;
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

    static this_t slerp(const this_t start, const this_t end, const scalar_type fraction, const scalar_type threshold = numeric_limits<scalar_type>::epsilon)
    {
        const scalar_type totalPseudoAngle = hlsl::dot(start.data, end.data);

        // make sure we use the short rotation
        const scalar_type cosA = ieee754::flipSignIfRHSNegative<scalar_type>(totalPseudoAngle, totalPseudoAngle);
        if (cosA <= (scalar_type(1.0) - threshold)) // spherical interpolation
        {
            assert(testing::relativeApproxCompare(hlsl::length(start.data), scalar_type(1.0), scalar_type(1e-4)));
            assert(testing::relativeApproxCompare(hlsl::length(end.data), scalar_type(1.0), scalar_type(1e-4)));

            this_t retval;
            const scalar_type sinARcp = scalar_type(1.0) / hlsl::sqrt(scalar_type(1.0) - cosA * cosA);
            const scalar_type sinAt = hlsl::sin(fraction * hlsl::acos(cosA));
            const scalar_type sinAt_over_sinA = sinAt*sinARcp;
            const scalar_type scale = hlsl::sqrt(scalar_type(1.0)-sinAt*sinAt) - sinAt_over_sinA*cosA; //cosAt-cos(A)sin(tA)/sin(A) = (sin(A)cos(tA)-cos(A)sin(tA))/sin(A)
            const data_type adjEnd = ieee754::flipSignIfRHSNegative<data_type,scalar_type>(end.data, totalPseudoAngle);
            retval.data = scale * start.data + sinAt_over_sinA * adjEnd;

            return retval;
        }
        else
            return unnormLerp(start, end, fraction, totalPseudoAngle);
    }

    this_t operator-() NBL_CONST_MEMBER_FUNC
    {
        this_t retval;
        retval.data.x = -data.x;
        retval.data.y = -data.y;
        retval.data.z = -data.z;
        retval.data.w = data.w;
        return retval;
    }

    data_type data;
};

}


namespace cpp_compat_intrinsics_impl
{
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
        retval.data.w = hlsl::sqrt(T(1.0) - hlsl::dot(q.data, q.data));
        return retval;
    }
};

template<typename T>
struct static_cast_helper<math::truncated_quaternion<T>, math::quaternion<T> >
{
    static inline math::truncated_quaternion<T> cast(const math::quaternion<T> q)
    {
        assert(testing::relativeApproxCompare(hlsl::length(q.data), T(1.0), T(1e-4)));
        math::truncated_quaternion<T> t;
        t.data.x = q.data.x;
        t.data.y = q.data.y;
        t.data.z = q.data.z;
        return t;
    }
};

template<typename T>
struct static_cast_helper<matrix<T,3,3>, math::quaternion<T> >
{
    static inline matrix<T,3,3> cast(const math::quaternion<T> q)
    {
        return q.__constructMatrix();
    }
};

template<typename T>
struct static_cast_helper<math::quaternion<T>, matrix<T,3,3> >
{
    static inline math::quaternion<T> cast(const matrix<T,3,3> m)
    {
        return math::quaternion<T>::create(m, true);
    }
};
}

}
}

#endif
