#ifndef _NBL_BUILTIN_HLSL_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_TRANSFORMATION_MATRIX_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T>
matrix<T, 4, 4> getMatrix3x4As4x4(const matrix<T, 3, 4>& mat)
{
	matrix<T, 4, 4> output;
	for (int i = 0; i < 3; ++i)
		output[i] = mat[i];
	output[3] = float32_t4(0.0f, 0.0f, 0.0f, 1.0f);

	return output;
}

// TODO: use portable_float when merged
//! multiplies matrices a and b, 3x4 matrices are treated as 4x4 matrices with 4th row set to (0, 0, 0 ,1)
template<typename T>
inline matrix<T, 3, 4> concatenateBFollowedByA(const matrix<T, 3, 4>& a, const matrix<T, 3, 4>& b)
{
	const auto a4x4 = getMatrix3x4As4x4(a);
	const auto b4x4 = getMatrix3x4As4x4(b);
	return matrix<T, 3, 4>(mul(a4x4, b4x4));
}

template<typename T>
inline matrix<T, 3, 4> buildCameraLookAtMatrixLH(
	const vector<T, 3>& position,
	const vector<T, 3>& target,
	const vector<T, 3>& upVector)
{
	const vector<T, 3> zaxis = core::normalize(target - position);
	const vector<T, 3> xaxis = core::normalize(core::cross(upVector, zaxis));
	const vector<T, 3> yaxis = core::cross(zaxis, xaxis);

	matrix<T, 3, 4> r;
	r[0] = vector<T, 3>(xaxis, -dot(xaxis, position));
	r[1] = vector<T, 3>(yaxis, -dot(yaxis, position));
	r[2] = vector<T, 3>(zaxis, -dot(zaxis, position));

	return r;
}

template<typename T>
inline matrix<T, 3, 4> buildCameraLookAtMatrixRH(
	const vector<T, 3>& position,
	const vector<T, 3>& target,
	const vector<T, 3>& upVector)
{
	const vector<T, 3> zaxis = core::normalize(position - target);
	const vector<T, 3> xaxis = core::normalize(core::cross(upVector, zaxis));
	const vector<T, 3> yaxis = core::cross(zaxis, xaxis);

	matrix<T, 3, 4> r;
	r[0] = vector<T, 3>(xaxis, -dot(xaxis, position));
	r[1] = vector<T, 3>(yaxis, -dot(yaxis, position));
	r[2] = vector<T, 3>(zaxis, -dot(zaxis, position));

	return r;
}

// TODO: test, check if there is better implementation
// TODO: move quaternion to nbl::hlsl
// TODO: why NBL_REF_ARG(MatType) doesn't work?????

//! Replaces curent rocation and scale by rotation represented by quaternion `quat`, leaves 4th row and 4th colum unchanged
template<typename T, uint32_t N>
inline void setRotation(matrix<T, N, 4>& outMat, NBL_CONST_REF_ARG(core::quaternion) quat)
{
	static_assert(N == 3 || N == 4);

	outMat[0] = vector<T, 4>(
		1 - 2 * (quat.y * quat.y + quat.z * quat.z),
		2 * (quat.x * quat.y - quat.z * quat.w),
		2 * (quat.x * quat.z + quat.y * quat.w),
		outMat[0][3]
	);

	outMat[1] = vector<T, 4>(
		2 * (quat.x * quat.y + quat.z * quat.w),
		1 - 2 * (quat.x * quat.x + quat.z * quat.z),
		2 * (quat.y * quat.z - quat.x * quat.w),
		outMat[1][3]
	);

	outMat[2] = vector<T, 4>(
		2 * (quat.x * quat.z - quat.y * quat.w),
		2 * (quat.y * quat.z + quat.x * quat.w),
		1 - 2 * (quat.x * quat.x + quat.y * quat.y),
		outMat[2][3]
	);
}

template<typename T, uint32_t N>
inline void setTranslation(matrix<T, N, 4>& outMat, NBL_CONST_REF_ARG(vector<T, 3>) translation)
{
	static_assert(N == 3 || N == 4);

	outMat[0].w = translation.x;
	outMat[1].w = translation.y;
	outMat[2].w = translation.z;
}

}
}

#endif