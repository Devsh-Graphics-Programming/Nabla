// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef __NBL_QUATERNION_H_INCLUDED__
#define __NBL_QUATERNION_H_INCLUDED__

#include "vectorSIMD.h"

#include "nbl/core/math/glslFunctions.h"

namespace nbl
{
namespace core
{
class matrix3x4SIMD;

//! Quaternion class for representing rotations.
/** It provides cheap combinations and avoids gimbal locks.
Also useful for interpolations. */
class quaternion : private vectorSIMDf
{
public:
    //! Default Constructor
    inline quaternion()
        : vectorSIMDf(0, 0, 0, 1) {}

    inline quaternion(const quaternion& other)
        : vectorSIMDf(static_cast<const vectorSIMDf&>(other)) {}

    inline quaternion(const float* data)
        : vectorSIMDf(data) {}

    //! Constructor
    inline quaternion(const float& x, const float& y, const float& z, const float& w)
        : vectorSIMDf(x, y, z, w) {}

    //! Constructor which converts euler angles (radians) to a quaternion
    inline quaternion(const float& pitch, const float& yaw, const float& roll) { set(pitch, yaw, roll); }

    //! Constructor which converts a matrix to a quaternion
    explicit quaternion(const matrix3x4SIMD& m);

    inline float* getPointer() { return pointer; }

    //! Equalilty operator
    inline vector4db_SIMD operator==(const quaternion& other) const { return vectorSIMDf::operator==(other); }

    //! inequality operator
    inline vector4db_SIMD operator!=(const quaternion& other) const { return vectorSIMDf::operator!=(other); }

    //! Assignment operator
    inline quaternion& operator=(const quaternion& other) { return reinterpret_cast<quaternion&>(vectorSIMDf::operator=(other)); }

    //! Multiplication operator with scalar
    inline quaternion operator*(const float& s) const
    {
        quaternion tmp;
        reinterpret_cast<vectorSIMDf&>(tmp) = reinterpret_cast<const vectorSIMDf*>(this)->operator*(s);
        return tmp;
    }

    //! Multiplication operator with scalar
    inline quaternion& operator*=(const float& s)
    {
        *this = (*this) * s;
        return *this;
    }

    //! Multiplication operator
    inline quaternion& operator*=(const quaternion& other)
    {
        *this = (*this) * other;
        return *this;
    }

    //! Multiplication operator
    //http://momchil-velikov.blogspot.fr/2013/10/fast-sse-quternion-multiplication.html
    inline quaternion operator*(const quaternion& other) const
    {
        __m128 xyzw = vectorSIMDf::getAsRegister();
        __m128 abcd = reinterpret_cast<const vectorSIMDf&>(other).getAsRegister();

        __m128 t0 = FAST_FLOAT_SHUFFLE(abcd, _MM_SHUFFLE(3, 3, 3, 3)); /* 1, 0.5 */
        __m128 t1 = FAST_FLOAT_SHUFFLE(xyzw, _MM_SHUFFLE(2, 3, 0, 1)); /* 1, 0.5 */

        __m128 t3 = FAST_FLOAT_SHUFFLE(abcd, _MM_SHUFFLE(0, 0, 0, 0)); /* 1, 0.5 */
        __m128 t4 = FAST_FLOAT_SHUFFLE(xyzw, _MM_SHUFFLE(1, 0, 3, 2)); /* 1, 0.5 */

        __m128 t5 = FAST_FLOAT_SHUFFLE(abcd, _MM_SHUFFLE(1, 1, 1, 1)); /* 1, 0.5 */
        __m128 t6 = FAST_FLOAT_SHUFFLE(xyzw, _MM_SHUFFLE(2, 0, 3, 1)); /* 1, 0.5 */

        /* [d,d,d,d]*[z,w,x,y] = [dz,dw,dx,dy] */
        __m128 m0 = _mm_mul_ps(t0, t1); /* 5/4, 1 */

        /* [a,a,a,a]*[y,x,w,z] = [ay,ax,aw,az]*/
        __m128 m1 = _mm_mul_ps(t3, t4); /* 5/4, 1 */

        /* [b,b,b,b]*[z,x,w,y] = [bz,bx,bw,by]*/
        __m128 m2 = _mm_mul_ps(t5, t6); /* 5/4, 1 */

        /* [c,c,c,c]*[w,z,x,y] = [cw,cz,cx,cy] */
        __m128 t7 = FAST_FLOAT_SHUFFLE(abcd, _MM_SHUFFLE(2, 2, 2, 2)); /* 1, 0.5 */
        __m128 t8 = FAST_FLOAT_SHUFFLE(xyzw, _MM_SHUFFLE(3, 2, 0, 1)); /* 1, 0.5 */

        __m128 m3 = _mm_mul_ps(t7, t8); /* 5/4, 1 */

        /* 1 */
        /* [dz,dw,dx,dy]+-[ay,ax,aw,az] = [dz+ay,dw-ax,dx+aw,dy-az] */
        __m128 e = _mm_addsub_ps(m0, m1); /* 3, 1 */

        /* 2 */
        /* [dx+aw,dz+ay,dy-az,dw-ax] */
        e = FAST_FLOAT_SHUFFLE(e, _MM_SHUFFLE(1, 3, 0, 2)); /* 1, 0.5 */

        /* [dx+aw,dz+ay,dy-az,dw-ax]+-[bz,bx,bw,by] = [dx+aw+bz,dz+ay-bx,dy-az+bw,dw-ax-by]*/
        e = _mm_addsub_ps(e, m2); /* 3, 1 */

        /* 2 */
        /* [dz+ay-bx,dw-ax-by,dy-az+bw,dx+aw+bz] */
        e = FAST_FLOAT_SHUFFLE(e, _MM_SHUFFLE(2, 0, 1, 3)); /* 1, 0.5 */

        /* [dz+ay-bx,dw-ax-by,dy-az+bw,dx+aw+bz]+-[cw,cz,cx,cy]
             = [dz+ay-bx+cw,dw-ax-by-cz,dy-az+bw+cx,dx+aw+bz-cy] */
        e = _mm_addsub_ps(e, m3); /* 3, 1 */

        /* 2 */
        /* [dw-ax-by-cz,dz+ay-bx+cw,dy-az+bw+cx,dx+aw+bz-cy] */
        quaternion tmp;
        reinterpret_cast<vectorSIMDf&>(tmp) = FAST_FLOAT_SHUFFLE(e, _MM_SHUFFLE(2, 3, 1, 0)); /* 1, 0.5 */
        return tmp;
    }

    inline vectorSIMDf transformVect(const vectorSIMDf& vec)
    {
        vectorSIMDf direction = *reinterpret_cast<const vectorSIMDf*>(this);
        vectorSIMDf scale = core::length(direction);
        direction.makeSafe3D();

        return scale * vec + cross(direction, vec * W + cross(direction, vec)) * 2.f;
    }

    //! Sets new quaternion
    inline quaternion& set(const vectorSIMDf& xyzw)
    {
        *this = reinterpret_cast<const quaternion&>(xyzw);
        return *this;
    }

    //! Sets new quaternion based on euler angles (radians)
    inline quaternion& set(const float& roll, const float& pitch, const float& yaw);

    //! Sets new quaternion from other quaternion
    inline quaternion& set(const quaternion& quat)
    {
        *this = quat;
        return *this;
    }

    //! Inverts this quaternion
    inline void makeInverse()
    {
        reinterpret_cast<vectorSIMDf&>(*this) ^= _mm_set_epi32(0x0u, 0x80000000u, 0x80000000u, 0x80000000u);
    }

    //! Fills an angle (radians) around an axis (unit vector)
    void toAngleAxis(float& angle, vector3df_SIMD& axis) const;

    //! Output this quaternion to an euler angle (radians)
    void toEuler(vector3df_SIMD& euler) const;

    //! Set quaternion to identity
    inline void makeIdentity() { vectorSIMDf::set(0, 0, 0, 1); }

    vectorSIMDf& getData() { return *((vectorSIMDf*)this); }

    //statics
    inline static quaternion normalize(const quaternion& in)
    {
        quaternion tmp;
        reinterpret_cast<vectorSIMDf&>(tmp) = core::normalize(reinterpret_cast<const vectorSIMDf&>(in));
        return tmp;
    }

    //! Helper func
    static quaternion lerp(const quaternion& q1, const quaternion& q2, const float& interpolant, const bool& wrongDoubleCover);

    //! Set this quaternion to the linear interpolation between two quaternions
    /** \param q1 First quaternion to be interpolated.
		\param q2 Second quaternion to be interpolated.
		\param interpolant Progress of interpolation. For interpolant=0 the result is
		q1, for interpolant=1 the result is q2. Otherwise interpolation
		between q1 and q2.
		*/
    static quaternion lerp(const quaternion& q1, const quaternion& q2, const float& interpolant);

    //! Helper func
    static inline void flerp_interpolant_terms(float& interpolantPrecalcTerm2, float& interpolantPrecalcTerm3, const float& interpolant)
    {
        interpolantPrecalcTerm2 = (interpolant - 0.5f) * (interpolant - 0.5f);
        interpolantPrecalcTerm3 = interpolant * (interpolant - 0.5f) * (interpolant - 1.f);
    }

    static float flerp_adjustedinterpolant(const float& angle, const float& interpolant, const float& interpolantPrecalcTerm2, const float& interpolantPrecalcTerm3);

    //! Set this quaternion to the approximate slerp between two quaternions
    /** \param q1 First quaternion to be interpolated.
		\param q2 Second quaternion to be interpolated.
		\param interpolant Progress of interpolation. For interpolant=0 the result is
		q1, for interpolant=1 the result is q2. Otherwise interpolation
		between q1 and q2.
		*/
    static quaternion flerp(const quaternion& q1, const quaternion& q2, const float& interpolant);

    //! Set this quaternion to the result of the spherical interpolation between two quaternions
    /** \param q1 First quaternion to be interpolated.
		\param q2 Second quaternion to be interpolated.
		\param time Progress of interpolation. For interpolant=0 the result is
		q1, for interpolant=1 the result is q2. Otherwise interpolation
		between q1 and q2.
		\param threshold To avoid inaccuracies the
		interpolation switches to linear interpolation at some point.
		This value defines how much of the interpolation will
		be calculated with lerp.
		*/
    static quaternion slerp(const quaternion& q1, const quaternion& q2,
        const float& interpolant, const float& threshold = .05f);

    inline static quaternion fromEuler(const vector3df_SIMD& euler)
    {
        quaternion tmp;
        tmp.set(euler.X, euler.Y, euler.Z);
        return tmp;
    }

    inline static quaternion fromEuler(const vector3df& euler)
    {
        quaternion tmp;
        tmp.set(euler.X, euler.Y, euler.Z);
        return tmp;
    }

    //! Set quaternion to represent a rotation from one vector to another.
    static quaternion rotationFromTo(const vector3df_SIMD& from, const vector3df_SIMD& to);

    //! Create quaternion from rotation angle and rotation axis.
    /** Axis must be unit length.
		The quaternion representing the rotation is
		q = cos(A/2)+sin(A/2)*(x*i+y*j+z*k).
		\param angle Rotation Angle in radians.
		\param axis Rotation axis. */
    static quaternion fromAngleAxis(const float& angle, const vector3df_SIMD& axis);
};
static_assert(sizeof(quaternion) == sizeof(vectorSIMDf), "Quaternion not same size as vec4");

// set this quaternion to the result of the linear interpolation between two quaternions
inline quaternion quaternion::lerp(const quaternion& q1, const quaternion& q2, const float& interpolant, const bool& wrongDoubleCover)
{
    vectorSIMDf retval;
    if(wrongDoubleCover)
        retval = mix<vectorSIMDf>(reinterpret_cast<const vectorSIMDf&>(q1), -reinterpret_cast<const vectorSIMDf&>(q2), vectorSIMDf(interpolant));
    else
        retval = mix<vectorSIMDf>(reinterpret_cast<const vectorSIMDf&>(q1), reinterpret_cast<const vectorSIMDf&>(q2), vectorSIMDf(interpolant));
    return reinterpret_cast<const quaternion&>(retval);
}

// set this quaternion to the result of the linear interpolation between two quaternions
inline quaternion quaternion::lerp(const quaternion& q1, const quaternion& q2, const float& interpolant)
{
    const float angle = dot<vectorSIMDf>(q1, q2)[0];
    return lerp(q1, q2, interpolant, angle < 0.0f);
}

// Arseny Kapoulkine
inline float quaternion::flerp_adjustedinterpolant(const float& angle, const float& interpolant, const float& interpolantPrecalcTerm2, const float& interpolantPrecalcTerm3)
{
    float A = 1.0904f + angle * (-3.2452f + angle * (3.55645f - angle * 1.43519f));
    float B = 0.848013f + angle * (-1.06021f + angle * 0.215638f);
    float k = A * interpolantPrecalcTerm2 + B;
    float ot = interpolant + interpolantPrecalcTerm3 * k;
    return ot;
}

// set this quaternion to the result of an approximate slerp
inline quaternion quaternion::flerp(const quaternion& q1, const quaternion& q2, const float& interpolant)
{
    const float angle = dot<vectorSIMDf>(q1, q2)[0];
    return lerp(q1, q2, flerp_adjustedinterpolant(fabsf(angle), interpolant, (interpolant - 0.5f) * (interpolant - 0.5f), interpolant * (interpolant - 0.5f) * (interpolant - 1.f)), angle < 0.0f);
}

// set this quaternion to the result of the interpolation between two quaternions
inline quaternion quaternion::slerp(const quaternion& q1, const quaternion& q2, const float& interpolant, const float& threshold)
{
    float angle = dot<vectorSIMDf>(q1, q2)[0];

    // make sure we use the short rotation
    bool wrongDoubleCover = angle < 0.0f;
    if(wrongDoubleCover)
        angle *= -1.f;

    if(angle <= (1.f - threshold))  // spherical interpolation
    {  // acosf + sinf
        vectorSIMDf retval;

        const float sinARcp = inversesqrt(1.f - angle * angle);
        const float sinAt = sinf(acosf(angle) * interpolant);  // could this line be optimized?
        //1sqrt 3min/add 5mul from now on
        const float sinAt_over_sinA = sinAt * sinARcp;

        const float scale = core::sqrt(1.f - sinAt * sinAt) - angle * sinAt_over_sinA;  //cosAt-cos(A)sin(tA)/sin(A) = (sin(A)cos(tA)-cos(A)sin(tA))/sin(A)
        if(wrongDoubleCover)  // make sure we use the short rotation
            retval = reinterpret_cast<const vectorSIMDf&>(q1) * scale - reinterpret_cast<const vectorSIMDf&>(q2) * sinAt_over_sinA;
        else
            retval = reinterpret_cast<const vectorSIMDf&>(q1) * scale + reinterpret_cast<const vectorSIMDf&>(q2) * sinAt_over_sinA;

        return reinterpret_cast<const quaternion&>(retval);
    }
    else
        return normalize(lerp(q1, q2, interpolant, wrongDoubleCover));
}

#if !NBL_TEST_BROKEN_QUATERNION_USE
//! axis must be unit length, angle in radians
inline quaternion quaternion::fromAngleAxis(const float& angle, const vector3df_SIMD& axis)
{
    const float fHalfAngle = 0.5f * angle;
    const float fSin = sinf(fHalfAngle);
    quaternion retval;
    reinterpret_cast<vectorSIMDf&>(retval) = axis * fSin;
    reinterpret_cast<vectorSIMDf&>(retval).W = cosf(fHalfAngle);
    return retval;
}

inline void quaternion::toAngleAxis(float& angle, vector3df_SIMD& axis) const
{
    vectorSIMDf scale = core::length(*reinterpret_cast<const vectorSIMDf*>(this));

    if(scale.X == 0.f)
    {
        angle = 0.0f;
        axis.X = 0.0f;
        axis.Y = 1.0f;
        axis.Z = 0.0f;
    }
    else
    {
        axis = reinterpret_cast<const vectorSIMDf*>(this)->operator/(scale);
        angle = 2.f * acosf(axis.W);

        axis.makeSafe3D();
    }
}

inline void quaternion::toEuler(vector3df_SIMD& euler) const
{
    vectorSIMDf sqr = *reinterpret_cast<const vectorSIMDf*>(this);
    sqr *= sqr;
    const double test = 2.0 * (Y * W - X * Z);

    if(core::equals(test, 1.0, 0.000001))
    {
        // heading = rotation about z-axis
        euler.Z = (float)(-2.0 * atan2(X, W));
        // bank = rotation about x-axis
        euler.X = 0;
        // attitude = rotation about y-axis
        euler.Y = core::HALF_PI<float>();
    }
    else if(core::equals(test, -1.0, 0.000001))
    {
        // heading = rotation about z-axis
        euler.Z = (float)(2.0 * atan2(X, W));
        // bank = rotation about x-axis
        euler.X = 0;
        // attitude = rotation about y-axis
        euler.Y = -core::HALF_PI<float>();
    }
    else
    {
        // heading = rotation about z-axis
        euler.Z = (float)atan2(2.0 * (X * Y + Z * W), (sqr.X - sqr.Y - sqr.Z + sqr.W));
        // bank = rotation about x-axis
        euler.X = (float)atan2(2.0 * (Y * Z + X * W), (-sqr.X - sqr.Y + sqr.Z + sqr.W));
        // attitude = rotation about y-axis
        euler.Y = (float)asin(core::clamp(test, -1.0, 1.0));
    }
}

inline quaternion quaternion::rotationFromTo(const vector3df_SIMD& from, const vector3df_SIMD& to)
{
    // Based on Stan Melax's article in Game Programming Gems
    // Copy, since cannot modify local
    vector3df_SIMD v0 = from;
    vector3df_SIMD v1 = to;
    v0 = core::normalize(v0);
    v1 = core::normalize(v1);

    const vectorSIMDf dddd = core::dot(v0, v1);
    quaternion tmp;
    if(dddd.X >= 1.0f)  // If dot == 1, vectors are the same
    {
        return tmp;
    }
    else if(dddd.X <= -1.0f)  // exactly opposite
    {
        vector3df_SIMD axis(1.0f, 0.f, 0.f);
        axis = cross(axis, v0);
        if(length(axis)[0] == 0.f)
        {
            axis.set(0.f, 1.f, 0.f);
            axis = cross(axis, v0);
        }
        // same as fromAngleAxis(PI, axis).normalize();
        reinterpret_cast<vectorSIMDf&>(tmp) = axis;
        return normalize(tmp);
    }

    vectorSIMDf s = core::sqrt(vectorSIMDf(2.f, 2.f, 2.f, 0.f) + dddd * 2.f);
    reinterpret_cast<vectorSIMDf&>(tmp) = cross(v0, v1) * reciprocal_approxim(s);
    tmp.W = s.X * 0.5f;
    return normalize(tmp);
}
#endif

// sets new quaternion based on euler angles
inline quaternion& quaternion::set(const float& roll, const float& pitch, const float& yaw)
{
    float angle;

    angle = roll * 0.5f;
    const float sr = sinf(angle);
    const float cr = cosf(angle);

    angle = pitch * 0.5f;
    const float sp = sinf(angle);
    const float cp = cos(angle);

    angle = yaw * 0.5f;
    const float sy = sinf(angle);
    const float cy = cosf(angle);

    const float cpcy = cp * cy;
    const float spcy = sp * cy;
    const float cpsy = cp * sy;
    const float spsy = sp * sy;

    *reinterpret_cast<vectorSIMDf*>(this) = vectorSIMDf(sr, cr, cr, cr) * vectorSIMDf(cpcy, spcy, cpsy, cpcy) + vectorSIMDf(-cr, sr, -sr, sr) * vectorSIMDf(spsy, cpsy, spcy, spsy);

    return *this;
}

}  // end namespace core
}  // end namespace nbl

#endif
