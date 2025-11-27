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

template<typename float_t>
struct quaternion
{
	// i*data[0] + j*data[1] + k*data[2] + data[3]
	using vec_t = vector<float_t, 4>;
	vector<float_t, 4> data;

	//! creates identity quaternion
	static inline quaternion create()
	{
		quaternion q;
		q.data = vector<float_t, 4>(0.0f, 0.0f, 0.0f, 1.0f);

		return q;
	}
	
	static inline quaternion create(float_t x, float_t y, float_t z, float_t w)
	{
		quaternion q;
		q.data = vector<float_t, 4>(x, y, z, w);

		return q;
	}

	static inline quaternion create(NBL_CONST_REF_ARG(quaternion) other)
	{
		return other;
	}

	static inline quaternion create(float_t pitch, float_t yaw, float_t roll)
	{
		const float rollDiv2 = roll * 0.5f;
		const float sr = sinf(rollDiv2);
		const float cr = cosf(rollDiv2);

		const float pitchDiv2 = pitch * 0.5f;
		const float sp = sinf(pitchDiv2);
		const float cp = cosf(pitchDiv2);

		const float yawDiv2 = yaw * 0.5f;
		const float sy = sinf(yawDiv2);
		const float cy = cosf(yawDiv2);

		quaternion<float_t> output;
		output.data[0] = cr * sp * cy + sr * cp * sy; // x
		output.data[1] = cr * cp * sy - sr * sp * cy; // y
		output.data[2] = sr * cp * cy - cr * sp * sy; // z
		output.data[3] = cr * cp * cy + sr * sp * sy; // w

		return output;
	}

	// TODO:
	//explicit quaternion(NBL_CONST_REF_ARG(float32_t3x4) m) {}

	inline quaternion operator*(float_t scalar)
	{
		quaternion output;
		output.data = data * scalar;
		return output;
	}

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

