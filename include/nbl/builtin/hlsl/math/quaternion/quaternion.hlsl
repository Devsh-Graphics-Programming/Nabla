// Copyright (C) 2019 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine" and was originally part of the "Irrlicht Engine"
// For conditions of distribution and use, see copyright notice in nabla.h
// See the original file in irrlicht source for authors

#ifndef _NBL_BUILTIN_HLSL_MATH_QUATERNION_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_QUATERNION_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

//! Quaternion class for representing rotations.
/** It provides cheap combinations and avoids gimbal locks.
Also useful for interpolations. */

template<typename T>
struct quaternion
{
	// i*data[0] + j*data[1] + k*data[2] + data[3]
	using vec_t = vector<T, 4>;
	vector<T, 4> data;

	//! creates identity quaternion
	static inline quaternion create()
	{
		quaternion q;
		q.data = vector<T, 4>(0.0f, 0.0f, 0.0f, 1.0f);

		return q;
	}
	
	static inline quaternion create(T x, T y, T z, T w)
	{
		quaternion q;
		q.data = vector<T, 4>(x, y, z, w);

		return q;
	}

	static inline quaternion create(NBL_CONST_REF_ARG(quaternion) other)
	{
		return other;
	}

	static inline quaternion create(T pitch, T yaw, T roll)
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

		quaternion<T> output;
		output.data = float32_t4(sr, cr, cr, cr) * float32_t4(cpcy, spcy, cpsy, cpcy) + float32_t4(-cr, sr, -sr, sr) * float32_t4(spsy, cpsy, spcy, spsy);

		return output;
	}

	// TODO:
	//explicit quaternion(NBL_CONST_REF_ARG(float32_t3x4) m) {}

#define DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR(TYPE)\
	inline quaternion operator*(TYPE scalar)\
	{\
		quaternion output;\
		output.data = data * scalar;\
		return output;\
	}\

	DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR(uint32_t)
	DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR(uint64_t)
	DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR(float32_t)
	DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR(float64_t)

#undef DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR

	inline quaternion operator*(NBL_CONST_REF_ARG(quaternion) other)
	{
		return quaternion::create(
			data.w * other.data.w - data.x * other.x - data.y * other.data.y - data.z * other.data.z,
			data.w * other.data.x + data.x * other.w + data.y * other.data.z - data.z * other.data.y,
			data.w * other.data.y - data.x * other.z + data.y * other.data.w + data.z * other.data.x,
			data.w * other.data.z + data.x * other.y - data.y * other.data.x + data.z * other.data.w
		);
	}
};

} // end namespace core
} // nbl

#endif

