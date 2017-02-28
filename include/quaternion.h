// Copyright (C) 2002-2012 Nikolaus Gebhardt
// This file is part of the "Irrlicht Engine".
// For conditions of distribution and use, see copyright notice in irrlicht.h
#ifndef __IRR_QUATERNION_H_INCLUDED__
#define __IRR_QUATERNION_H_INCLUDED__



#include "irrTypes.h"
#include "irrMath.h"
#include "matrix4.h"
#include "matrix4x3.h"
#include "vector3d.h"
#include "vectorSIMD.h"

// Between Irrlicht 1.7 and Irrlicht 1.8 the quaternion-matrix conversions got fixed.
// This define disables all involved functions completely to allow finding all places
// where the wrong conversions had been in use.
///#define IRR_TEST_BROKEN_QUATERNION_USE 1

namespace irr
{
namespace core
{


//! Quaternion class for representing rotations.
/** It provides cheap combinations and avoids gimbal locks.
Also useful for interpolations. */
class quaternion : private vectorSIMDf
{
	public:
        static inline void* operator new(size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void operator delete(void* ptr)
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new[](size_t size) throw(std::bad_alloc)
        {
            void *memoryallocatedaligned = 0;
#ifdef _IRR_WINDOWS_
            memoryallocatedaligned = _aligned_malloc(size,SIMD_ALIGNMENT);
#else
            posix_memalign((void**)&memoryallocatedaligned,SIMD_ALIGNMENT,size);
#endif
            return memoryallocatedaligned;
        }
        static inline void  operator delete[](void* ptr) throw()
        {
#ifdef _IRR_WINDOWS_
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
        static inline void* operator new(std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete(void* p,void* t) throw() {}
        static inline void* operator new[](std::size_t size,void* p) throw(std::bad_alloc)
        {
            return p;
        }
        static inline void  operator delete[](void* p,void* t) throw() {}



		//! Default Constructor
		inline quaternion() : vectorSIMDf(0,0,0,1) {}

		inline quaternion(const float* data) : vectorSIMDf(data) {}

		//! Constructor
		inline quaternion(const float& x, const float& y, const float& z, const float& w) : vectorSIMDf(x,y,z,w) { }

		//! Constructor which converts euler angles (radians) to a quaternion
		inline quaternion(const float& pitch, const float& yaw, const float& roll) {set(pitch,yaw,roll);}

		//! Constructor which converts a matrix to a quaternion
		inline quaternion(const matrix4x3& mat) {*this = mat;}

        inline float* getPointer() {return pointer;}

		//! Equalilty operator
		inline vector4db_SIMD operator==(const quaternion& other) const {return vectorSIMDf::operator==(other);}

		//! inequality operator
		inline vector4db_SIMD operator!=(const quaternion& other) const {return vectorSIMDf::operator!=(other);}

		//! Assignment operator
		inline quaternion& operator=(const quaternion& other) {return reinterpret_cast<quaternion&>(vectorSIMDf::operator=(other));}

		//! Matrix assignment operator
		inline quaternion& operator=(const matrix4x3& m)
        {
/*
            __m128 one = _mm_set1_ps(1.f);
            __m128 Qx  = _mm_xor_ps(_mm_load1_ps(&m(0,0)),_mm_castsi128_ps(_mm_set_epi32(0x80000000u,0x80000000u,0,0)));
            __m128 Qy  = _mm_xor_ps(_mm_load1_ps(&m(1,1)),_mm_castsi128_ps(_mm_set_epi32(0x80000000u,0,0x80000000u,0)));
            __m128 Qz  = _mm_xor_ps(_mm_load1_ps(&m(2,2)),_mm_castsi128_ps(_mm_set_epi32(0,0x80000000u,0x80000000u,0)));

            __m128 output = _mm_add_ps(_mm_add_ps(one,Qx),_mm_add_ps(Qy,Qz));
            output = _mm_and_ps(_mm_mul_ps(_mm_sqrt_ps(output),_mm_set1_ps(0.5f)),_mm_castsi128_ps(_mm_set_epi32(0x7fffffffu,0x7fffffffu,0x7fffffffu,0xffffffffu)));

            __m128 mask = _mm_and_ps(_mm_cmplt_ps(_mm_set_ps(m(1,0),m(0,2),m(2,1),0.f),_mm_set_ps(m(0,1),m(2,0),m(1,2),0.f)),_mm_castsi128_ps(_mm_set_epi32(0x80000000u,0x80000000u,0x80000000u,0)));

            _mm_store_ps(pointer,_mm_or_ps(output,mask));
*/

            const float diag = m(0,0) + m(1,1) + m(2,2);

            if( diag > -1.0f )
            {
                const float scale = sqrtf(diag+1.f) * 2.0f; // get scale from diagonal

                // TODO: speed this up
                X = (m(2,1) - m(1,2)) / scale;
                Y = (m(0,2) - m(2,0)) / scale;
                Z = (m(1,0) - m(0,1)) / scale;
                W = 0.25f * scale;
            }
            else
            {
                if (m(0,0)>m(1,1) && m(0,0)>m(2,2))
                {
                    // 1st element of diag is greatest value
                    // find scale according to 1st element, and double it
                    const float scale = sqrtf(1.0f + m(0,0) - m(1,1) - m(2,2)) * 2.0f;

                    // TODO: speed this up
                    X = 0.25f * scale;
                    Y = (m(0,1) + m(1,0)) / scale;
                    Z = (m(2,0) + m(0,2)) / scale;
                    W = (m(2,1) - m(1,2)) / scale;
                }
                else if (m(1,1)>m(2,2))
                {
                    // 2nd element of diag is greatest value
                    // find scale according to 2nd element, and double it
                    const float scale = sqrtf(1.0f + m(1,1) - m(0,0) - m(2,2)) * 2.0f;

                    // TODO: speed this up
                    X = (m(0,1) + m(1,0)) / scale;
                    Y = 0.25f * scale;
                    Z = (m(1,2) + m(2,1)) / scale;
                    W = (m(0,2) - m(2,0)) / scale;
                }
                else
                {
                    // 3rd element of diag is greatest value
                    // find scale according to 3rd element, and double it
                    const float scale = sqrtf(1.0f + m(2,2) - m(0,0) - m(1,1)) * 2.0f;

                    // TODO: speed this up
                    X = (m(0,2) + m(2,0)) / scale;
                    Y = (m(1,2) + m(2,1)) / scale;
                    Z = 0.25f * scale;
                    W = (m(1,0) - m(0,1)) / scale;
                }
            }
            *this = normalize(*this);
            return *this;
        }

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
		    *this = (*this)*s;
		    return *this;
		}

		//! Multiplication operator
		inline quaternion& operator*=(const quaternion& other)
		{
		    *this = (*this)*other;
		    return *this;
		}

		//! Multiplication operator
		//http://momchil-velikov.blogspot.fr/2013/10/fast-sse-quternion-multiplication.html
		inline quaternion operator*(const quaternion& other) const
        {
            __m128 xyzw = vectorSIMDf::getAsRegister();
            __m128 abcd = reinterpret_cast<const vectorSIMDf&>(other).getAsRegister();

          __m128 t0 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(abcd), _MM_SHUFFLE (3, 3, 3, 3))); /* 1, 0.5 */
          __m128 t1 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xyzw), _MM_SHUFFLE (2, 3, 0, 1))); /* 1, 0.5 */

          __m128 t3 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(abcd), _MM_SHUFFLE (0, 0, 0, 0))); /* 1, 0.5 */
          __m128 t4 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xyzw), _MM_SHUFFLE (1, 0, 3, 2))); /* 1, 0.5 */

          __m128 t5 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(abcd), _MM_SHUFFLE (1, 1, 1, 1))); /* 1, 0.5 */
          __m128 t6 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xyzw), _MM_SHUFFLE (2, 0, 3, 1))); /* 1, 0.5 */

          /* [d,d,d,d]*[z,w,x,y] = [dz,dw,dx,dy] */
          __m128 m0 = _mm_mul_ps (t0, t1); /* 5/4, 1 */

          /* [a,a,a,a]*[y,x,w,z] = [ay,ax,aw,az]*/
          __m128 m1 = _mm_mul_ps (t3, t4); /* 5/4, 1 */

          /* [b,b,b,b]*[z,x,w,y] = [bz,bx,bw,by]*/
          __m128 m2 = _mm_mul_ps (t5, t6); /* 5/4, 1 */

          /* [c,c,c,c]*[w,z,x,y] = [cw,cz,cx,cy] */
          __m128 t7 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(abcd), _MM_SHUFFLE (2, 2, 2, 2))); /* 1, 0.5 */
          __m128 t8 = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(xyzw), _MM_SHUFFLE (3, 2, 0, 1))); /* 1, 0.5 */

          __m128 m3 = _mm_mul_ps (t7, t8); /* 5/4, 1 */

          /* 1 */
          /* [dz,dw,dx,dy]+-[ay,ax,aw,az] = [dz+ay,dw-ax,dx+aw,dy-az] */
          __m128 e = _mm_addsub_ps (m0, m1); /* 3, 1 */

          /* 2 */
          /* [dx+aw,dz+ay,dy-az,dw-ax] */
          e = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(e), _MM_SHUFFLE (1, 3, 0, 2))); /* 1, 0.5 */

          /* [dx+aw,dz+ay,dy-az,dw-ax]+-[bz,bx,bw,by] = [dx+aw+bz,dz+ay-bx,dy-az+bw,dw-ax-by]*/
          e = _mm_addsub_ps (e, m2); /* 3, 1 */

          /* 2 */
          /* [dz+ay-bx,dw-ax-by,dy-az+bw,dx+aw+bz] */
          e = _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(e), _MM_SHUFFLE (2, 0, 1, 3))); /* 1, 0.5 */

          /* [dz+ay-bx,dw-ax-by,dy-az+bw,dx+aw+bz]+-[cw,cz,cx,cy]
             = [dz+ay-bx+cw,dw-ax-by-cz,dy-az+bw+cx,dx+aw+bz-cy] */
          e = _mm_addsub_ps (e, m3); /* 3, 1 */

          /* 2 */
          /* [dw-ax-by-cz,dz+ay-bx+cw,dy-az+bw+cx,dx+aw+bz-cy] */
          quaternion tmp;
          reinterpret_cast<vectorSIMDf&>(tmp) = vectorSIMDf(_mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(e), _MM_SHUFFLE (2, 3, 1, 0)))); /* 1, 0.5 */
          return tmp;
        }

        inline vectorSIMDf transformVect(const vectorSIMDf& vec)
        {
            vectorSIMDf direction = *reinterpret_cast<const vectorSIMDf*>(this);
            vectorSIMDf scale = direction.getLength();
            direction.makeSafe3D();

            return scale*vec+direction.crossProduct(vec*W+direction.crossProduct(vec))*2.f;
        }

		//! Calculates the dot product
		inline vectorSIMDf dotProduct(const quaternion& other) const
		{
		    return vectorSIMDf::dotProduct(reinterpret_cast<const vectorSIMDf&>(other));
		}
		inline float dotProductAsFloat(const quaternion& other) const
		{
		    return vectorSIMDf::dotProductAsFloat(reinterpret_cast<const vectorSIMDf&>(other));
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

		//! Creates a matrix from this quaternion
        inline matrix4x3 getMatrix() const
        {
            matrix4x3 m;
            getMatrix(m);
            return m;
        }

		//! Creates a matrix from this quaternion
		void getMatrix( matrix4x3 &dest, const vector3df_SIMD &translation=vectorSIMDf() ) const;
		void getMatrix_Sub3x3Transposed( matrix4x3 &dest, const vector3df_SIMD &translation=vectorSIMDf() ) const;

		/*!
			Creates a matrix from this quaternion
			Rotate about a center point
			shortcut for
			quaternion q;
			q.rotationFromTo ( vin[i].Normal, forward );
			q.getMatrixCenter ( lookat, center, newPos );

			matrix4x3 m2;
			m2.setInverseTranslation ( center );
			lookat *= m2;

			matrix4x3 m3;
			m2.setTranslation ( newPos );
			lookat *= m3;

		*/
		void getMatrixCenter( matrix4x3 &dest, const vector3df_SIMD &center, const vector3df_SIMD &translation ) const;

		//! Inverts this quaternion
		inline void makeInverse()
		{
		    __m128 result = _mm_xor_ps(reinterpret_cast<vectorSIMDf*>(this)->getAsRegister(),_mm_castsi128_ps(_mm_set_epi32(0x0u,0x80000000u,0x80000000u,0x80000000u)));
		    this->set(result);
		}

		//! Fills an angle (radians) around an axis (unit vector)
		void toAngleAxis(float& angle, vector3df_SIMD& axis) const;

		//! Output this quaternion to an euler angle (radians)
		void toEuler(vector3df_SIMD& euler) const;

		//! Set quaternion to identity
		inline void makeIdentity() {vectorSIMDf::set(0,0,0,1);}


        vectorSIMDf& getData() {return *((vectorSIMDf*)this);}

//statics
        inline static quaternion normalize(const quaternion& in)
        {
            quaternion tmp;
            reinterpret_cast<vectorSIMDf&>(tmp) = core::normalize(reinterpret_cast<const vectorSIMDf&>(in));
            return tmp;
        }

		//! Set this quaternion to the linear interpolation between two quaternions
		/** \param q1 First quaternion to be interpolated.
		\param q2 Second quaternion to be interpolated.
		\param time Progress of interpolation. For time=0 the result is
		q1, for time=1 the result is q2. Otherwise interpolation
		between q1 and q2.
		*/
		static quaternion lerp(const quaternion &q1, const quaternion &q2, const float& interpolant);

		//! Set this quaternion to the result of the spherical interpolation between two quaternions
		/** \param q1 First quaternion to be interpolated.
		\param q2 Second quaternion to be interpolated.
		\param time Progress of interpolation. For time=0 the result is
		q1, for time=1 the result is q2. Otherwise interpolation
		between q1 and q2.
		\param threshold To avoid inaccuracies at the end (time=1) the
		interpolation switches to linear interpolation at some point.
		This value defines how much of the remaining interpolation will
		be calculated with lerp. Everything from 1-threshold up will be
		linear interpolation.
		*/
		static quaternion slerp(quaternion q1, const quaternion& q2,
				const float& interpolant, const float& threshold=.05f);

		inline static quaternion fromEuler(const vector3df_SIMD& euler)
		{
		    quaternion tmp;
		    tmp.set(euler.X,euler.Y,euler.Z);
		    return tmp;
        }

		inline static quaternion fromEuler(const vector3df& euler)
		{
		    quaternion tmp;
		    tmp.set(euler.X,euler.Y,euler.Z);
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



/*!
	Creates a matrix from this quaternion
*/
inline void quaternion::getMatrix(matrix4x3 &dest,
		const core::vector3df_SIMD &center) const
{
	dest(0,0) = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
	dest(1,0) = 2.0f*X*Y + 2.0f*Z*W;
	dest(2,0) = 2.0f*X*Z - 2.0f*Y*W;

	dest(0,1) = 2.0f*X*Y - 2.0f*Z*W;
	dest(1,1) = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
	dest(2,1) = 2.0f*Z*Y + 2.0f*X*W;

	dest(0,2) = 2.0f*X*Z + 2.0f*Y*W;
	dest(1,2) = 2.0f*Z*Y - 2.0f*X*W;
	dest(2,2) = 1.0f - 2.0f*X*X - 2.0f*Y*Y;

	dest(0,3) = center.X;
	dest(1,3) = center.Y;
	dest(2,3) = center.Z;
}

inline void quaternion::getMatrix_Sub3x3Transposed(matrix4x3 &dest,
		const core::vector3df_SIMD &center) const
{
	dest(0,0) = 1.0f - 2.0f*Y*Y - 2.0f*Z*Z;
	dest(0,1) = 2.0f*X*Y + 2.0f*Z*W;
	dest(0,2) = 2.0f*X*Z - 2.0f*Y*W;

	dest(1,0) = 2.0f*X*Y - 2.0f*Z*W;
	dest(1,1) = 1.0f - 2.0f*X*X - 2.0f*Z*Z;
	dest(1,2) = 2.0f*Z*Y + 2.0f*X*W;

	dest(2,0) = 2.0f*X*Z + 2.0f*Y*W;
	dest(2,1) = 2.0f*Z*Y - 2.0f*X*W;
	dest(2,2) = 1.0f - 2.0f*X*X - 2.0f*Y*Y;

	dest(0,3) = center.X;
	dest(1,3) = center.Y;
	dest(2,3) = center.Z;
}

/*!
	Creates a matrix from this quaternion
	Rotate about a center point
	shortcut for
	quaternion q;
	q.rotationFromTo(vin[i].Normal, forward);
	q.getMatrix(lookat, center);

	matrix4x3 m2;
	m2.setInverseTranslation(center);
	lookat *= m2;
*/
inline void quaternion::getMatrixCenter(matrix4x3 &dest,
					const vector3df_SIMD &center,
					const vector3df_SIMD &translation) const
{
    getMatrix(dest);

	dest.setRotationCenter ( center.getAsVector3df(), translation.getAsVector3df() );
}


// set this quaternion to the result of the linear interpolation between two quaternions
inline quaternion quaternion::lerp(const quaternion &q1, const quaternion &q2, const float& interpolant)
{
	vectorSIMDf tmp = reinterpret_cast<const vectorSIMDf&>(q1) + (reinterpret_cast<const vectorSIMDf&>(q2)-reinterpret_cast<const vectorSIMDf&>(q1))*interpolant;
	return reinterpret_cast<quaternion&>(tmp);
}


#if !IRR_TEST_BROKEN_QUATERNION_USE
// set this quaternion to the result of the interpolation between two quaternions
inline quaternion quaternion::slerp(quaternion q1, const quaternion &q2, const float& interpolant, const float& threshold)
{
	vectorSIMDf angle = q1.dotProduct(q2);

	// make sure we use the short rotation
	if (angle.X < 0.0f)
	{
		q1 *= -1.0f;
		angle *= -1.0f;
	}

	if (angle.X <= (1.f-threshold)) // spherical interpolation
	{
		const f32 theta = acosf(angle.X);
		const vectorSIMDf sintheta(sinf(theta));
        const vectorSIMDf scale(sinf(theta * (1.0f-interpolant)));
		const vectorSIMDf invscale(sinf(theta * interpolant));
		vectorSIMDf retval = (reinterpret_cast<vectorSIMDf&>(q1)*scale + reinterpret_cast<const vectorSIMDf&>(q2)*invscale)/sintheta;
		return reinterpret_cast<quaternion&>(retval);
	}
	else // linear interploation
		return lerp(q1,q2,interpolant);
}


//! axis must be unit length, angle in radians
inline quaternion quaternion::fromAngleAxis(const float& angle, const vector3df_SIMD& axis)
{
	const f32 fHalfAngle = 0.5f*angle;
	const f32 fSin = sinf(fHalfAngle);
	quaternion retval;
	reinterpret_cast<vectorSIMDf&>(retval) = axis*fSin;
	reinterpret_cast<vectorSIMDf&>(retval).W = cosf(fHalfAngle);
	return retval;
}


inline void quaternion::toAngleAxis(float& angle, vector3df_SIMD &axis) const
{
    vectorSIMDf scale = reinterpret_cast<const vectorSIMDf*>(this)->getLength();

	if (scale.X==0.f)
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
	const f64 sqw = W*W;
	const f64 sqx = X*X;
	const f64 sqy = Y*Y;
	const f64 sqz = Z*Z;
	vectorSIMDf sqr = *reinterpret_cast<const vectorSIMDf*>(this);
	sqr *= sqr;
	const f64 test = 2.0 * (Y*W - X*Z);

	if (core::equals(test, 1.0, 0.000001))
	{
		// heading = rotation about z-axis
		euler.Z = (f32) (-2.0*atan2(X, W));
		// bank = rotation about x-axis
		euler.X = 0;
		// attitude = rotation about y-axis
		euler.Y = (f32) (PI64/2.0);
	}
	else if (core::equals(test, -1.0, 0.000001))
	{
		// heading = rotation about z-axis
		euler.Z = (f32) (2.0*atan2(X, W));
		// bank = rotation about x-axis
		euler.X = 0;
		// attitude = rotation about y-axis
		euler.Y = (f32) (PI64/-2.0);
	}
	else
	{
		// heading = rotation about z-axis
		euler.Z = (f32) atan2(2.0 * (X*Y +Z*W),(sqr.X - sqr.Y - sqr.Z + sqr.W));
		// bank = rotation about x-axis
		euler.X = (f32) atan2(2.0 * (Y*Z +X*W),(-sqr.X - sqr.Y + sqr.Z + sqr.W));
		// attitude = rotation about y-axis
		euler.Y = (f32) asin( core::clamp(test, -1.0, 1.0) );
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

	const vectorSIMDf dddd = v0.dotProduct(v1);
	quaternion tmp;
	if (dddd.X >= 1.0f) // If dot == 1, vectors are the same
	{
		return tmp;
	}
	else if (dddd.X <= -1.0f) // exactly opposite
	{
		vector3df_SIMD axis(1.0f, 0.f, 0.f);
		axis = axis.crossProduct(v0);
		if (axis.getLengthAsFloat()==0.f)
		{
			axis.set(0.f,1.f,0.f);
			axis = axis.crossProduct(v0);
		}
		// same as fromAngleAxis(PI, axis).normalize();
		reinterpret_cast<vectorSIMDf&>(tmp) = axis;
		return normalize(tmp);
	}

    vectorSIMDf s = sqrt(vectorSIMDf(2.f,2.f,2.f,0.f)+dddd*2.f);
	reinterpret_cast<vectorSIMDf&>(tmp) = v0.crossProduct(v1)*reciprocal(s);
	tmp.W = s.X*0.5f;
    return normalize(tmp);
}
#endif

// sets new quaternion based on euler angles
inline quaternion& quaternion::set(const float& roll, const float& pitch, const float& yaw)
{
	f64 angle;

	angle = roll * 0.5;
	const f64 sr = sin(angle);
	const f64 cr = cos(angle);

	angle = pitch * 0.5;
	const f64 sp = sin(angle);
	const f64 cp = cos(angle);

	angle = yaw * 0.5;
	const f64 sy = sin(angle);
	const f64 cy = cos(angle);

	const f64 cpcy = cp * cy;
	const f64 spcy = sp * cy;
	const f64 cpsy = cp * sy;
	const f64 spsy = sp * sy;

    *reinterpret_cast<vectorSIMDf*>(this) = vectorSIMDf(sr,cr,cr,cr)*vectorSIMDf(cpcy,spcy,cpsy,cpcy)+vectorSIMDf(-cr,sr,-sr,sr)*vectorSIMDf(spsy,cpsy,spcy,spsy);

	return *this;
}

} // end namespace core
} // end namespace irr

#endif // __IRR_QUATERNION_H_INCLUDED__

