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
		q.data = vector<T, 4>(x, y, z, z);

		return q;
	}

#define DEFINE_MUL_QUATERNION_BY_SCALAR_OPERATOR(TYPE)\
	inline quaternion operator*(float scalar)\
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
			w * q.w - x * q.x - y * q.y - z * q.z,
			w * q.x + x * q.w + y * q.z - z * q.y,
			w * q.y - x * q.z + y * q.w + z * q.x,
			w * q.z + x * q.y - y * q.x + z * q.w
		);
	}
}

} // end namespace core
} // nbl

#endif

