#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/quaternion/quaternion.hlsl>
// TODO: remove this header when deleting vectorSIMDf.hlsl
#include <nbl/core/math/glslFunctions.h>
#include "vectorSIMD.h"

namespace nbl
{
namespace hlsl
{

// TODO: if `IdentityFloat32_t3x4` and `IdentityFloat32_t3x4` constexprs are ok, then I can expand them into templated struct, not doing it untill the concept is approved
//template<typename T, uint32_t N, uint32_t M>
//struct IdentityMatrix
//{
//
//};
NBL_CONSTEXPR hlsl::float32_t3x4 IdentityFloat32_t3x4 =
	hlsl::float32_t3x4(hlsl::float32_t4(1, 0, 0, 0), hlsl::float32_t4(0, 0, 1, 0), hlsl::float32_t4(0, 0, 1, 0));
NBL_CONSTEXPR hlsl::float32_t4x4 IdentityFloat32_t4x4 =
	hlsl::float32_t4x4(hlsl::float32_t4(1, 0, 0, 0), hlsl::float32_t4(0, 0, 1, 0), hlsl::float32_t4(0, 0, 1, 0), hlsl::float32_t4(0, 0, 0, 1));

// TODO: this is temporary function, delete when removing vectorSIMD
template<typename T>
inline core::vectorSIMDf transformVector(const matrix<T, 4, 4>& mat, const core::vectorSIMDf& vec)
{
	core::vectorSIMDf output;
	float32_t4 tmp;
	for (int i = 0; i < 4; ++i) // rather do that that reinterpret_cast for safety
		tmp[i] = output[i];

	for (int i = 0; i < 4; ++i)
		output[i] = hlsl::dot<float32_t4>(mat[i], tmp);

	return output;
}

template<typename T>
inline matrix<T, 4, 4> getMatrix3x4As4x4(const matrix<T, 3, 4>& mat)
{
	matrix<T, 4, 4> output;
	for (int i = 0; i < 3; ++i)
		output[i] = mat[i];
	output[3] = float32_t4(0.0f, 0.0f, 0.0f, 1.0f);

	return output;
}

template<typename T, uint32_t N>
inline matrix<T, 3, 3> getSub3x3(const matrix<T, N, 4>& mat)
{
	return matrix<T, 3, 3>(mat);
}

template<uint32_t N, uint32_t M>
inline matrix<float64_t, N, M> getAs64BitPrecisionMatrix(const matrix<float32_t, N, M>& mat)
{
	matrix<float64_t, N, M> output;
	for (int i = 0; i < N; ++i)
		output[i] = mat[i];

	return output;
}

namespace transformation_matrix_utils_impl
{
	template<typename T>
	inline T determinant_helper(const matrix<T, 3, 3>& mat, vector<T, 3>& r1crossr2)
	{
		r1crossr2 = hlsl::cross(mat[1], mat[2]);
		return hlsl::dot(mat[0], r1crossr2);
	}
}

template<typename T, uint32_t N, uint32_t M>
inline matrix<T, 3, 3> getSub3x3TransposeCofactors(const matrix<T, N, M>& mat)
{
	static_assert(N >= 3 && M >= 3);

	matrix<T, 3, 3> output;
	vector<T, 3> row0 = vector<T, 3>(mat[0]);
	vector<T, 3> row1 = vector<T, 3>(mat[1]);
	vector<T, 3> row2 = vector<T, 3>(mat[2]);
	output[0] = hlsl::cross(row1, row2);
	output[1] = hlsl::cross(row2, row0);
	output[2] = hlsl::cross(row0, row1);

	output[0] = hlsl::cross(row0, row1);

	return output;
}

template<typename T, uint32_t N>
inline bool getSub3x3InverseTranspose(const matrix<T, N, 4>& matIn, matrix<T, 3, 3>& matOut)
{
	matrix<T, 3, 3> matIn3x3 = getSub3x3(matIn);
	vector<T, 3> r1crossr2;
	T d = transformation_matrix_utils_impl::determinant_helper(matIn3x3, r1crossr2);
	if (core::iszero(d, FLT_MIN))
		return false;
	auto rcp = core::reciprocal(d);

	// matrix of cofactors * 1/det
	matOut = getSub3x3TransposeCofactors(matIn3x3);
	matOut[0] *= rcp;
	matOut[1] *= rcp;
	matOut[2] *= rcp;

	return true;
}

// TODO: use portable_float when merged
//! multiplies matrices a and b, 3x4 matrices are treated as 4x4 matrices with 4th row set to (0, 0, 0 ,1)
template<typename T>
inline matrix<T, 3, 4> concatenateBFollowedByA(const matrix<T, 3, 4>& a, const matrix<T, 3, 4>& b)
{
	const matrix<T, 4, 4> a4x4 = getMatrix3x4As4x4(a);
	const matrix<T, 4, 4> b4x4 = getMatrix3x4As4x4(b);
	return matrix<T, 3, 4>(mul(a4x4, b4x4));
}

// /Arek: glm:: for normalize till dot product is fixed (ambiguity with glm namespace + linker issues)
template<typename T>
inline matrix<T, 3, 4> buildCameraLookAtMatrixLH(
	const vector<T, 3>& position,
	const vector<T, 3>& target,
	const vector<T, 3>& upVector)
{
	const vector<T, 3> zaxis = hlsl::normalize(target - position);
	const vector<T, 3> xaxis = hlsl::normalize(hlsl::cross(upVector, zaxis));
	const vector<T, 3> yaxis = hlsl::cross(zaxis, xaxis);

	matrix<T, 3, 4> r;
	r[0] = vector<T, 4>(xaxis, -hlsl::dot(xaxis, position));
	r[1] = vector<T, 4>(yaxis, -hlsl::dot(yaxis, position));
	r[2] = vector<T, 4>(zaxis, -hlsl::dot(zaxis, position));

	return r;
}

template<typename T>
inline matrix<T, 3, 4> buildCameraLookAtMatrixRH(
	const vector<T, 3>& position,
	const vector<T, 3>& target,
	const vector<T, 3>& upVector)
{
	const vector<T, 3> zaxis = hlsl::normalize(position - target);
	const vector<T, 3> xaxis = hlsl::normalize(hlsl::cross(upVector, zaxis));
	const vector<T, 3> yaxis = hlsl::cross(zaxis, xaxis);

	matrix<T, 3, 4> r;
	r[0] = vector<T, 4>(xaxis, -hlsl::dot(xaxis, position));
	r[1] = vector<T, 4>(yaxis, -hlsl::dot(yaxis, position));
	r[2] = vector<T, 4>(zaxis, -hlsl::dot(zaxis, position));

	return r;
}

// TODO: test, check if there is better implementation
// TODO: move quaternion to nbl::hlsl
// TODO: why NBL_REF_ARG(MatType) doesn't work?????

template<typename T, uint32_t N>
inline void setScale(matrix<T, N, 4>& outMat, NBL_CONST_REF_ARG(vector<T, 3>) scale)
{
	outMat[0][0] = scale[0];
	outMat[1][1] = scale[1];
	outMat[2][2] = scale[2];
}

//! Replaces curent rocation and scale by rotation represented by quaternion `quat`, leaves 4th row and 4th colum unchanged
template<typename T, uint32_t N>
inline void setRotation(matrix<T, N, 4>& outMat, NBL_CONST_REF_ARG(nbl::hlsl::quaternion<T>) quat)
{
	static_assert(N == 3 || N == 4);

	outMat[0] = vector<T, 4>(
		1 - 2 * (quat.data.y * quat.data.y + quat.data.z * quat.data.z),
		2 * (quat.data.x * quat.data.y - quat.data.z * quat.data.w),
		2 * (quat.data.x * quat.data.z + quat.data.y * quat.data.w),

		outMat[0][3]
	);

	outMat[1] = vector<T, 4>(
		2 * (quat.data.x * quat.data.y + quat.data.z * quat.data.w),
		1 - 2 * (quat.data.x * quat.data.x + quat.data.z * quat.data.z),
		2 * (quat.data.y * quat.data.z - quat.data.x * quat.data.w),
		outMat[1][3]
	);

	outMat[2] = vector<T, 4>(
		2 * (quat.data.x * quat.data.z - quat.data.y * quat.data.w),
		2 * (quat.data.y * quat.data.z + quat.data.x * quat.data.w),
		1 - 2 * (quat.data.x * quat.data.x + quat.data.y * quat.data.y),
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


template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixPerspectiveFovRH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians * 0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(w, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -h, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, -zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, -1.f, 0.f);

	return m;
}
template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixPerspectiveFovLH(float fieldOfViewRadians, float aspectRatio, float zNear, float zFar)
{
	const float h = core::reciprocal<float>(tanf(fieldOfViewRadians * 0.5f));
	_NBL_DEBUG_BREAK_IF(aspectRatio == 0.f); //division by zero
	const float w = h / aspectRatio;

	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(w, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -h, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, zFar / (zFar - zNear), -zNear * zFar / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, 1.f, 0.f);

	return m;
}

template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixOrthoRH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, -1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, 0.f, 1.f);

	return m;
}

template<typename T>
inline matrix<T, 4, 4> buildProjectionMatrixOrthoLH(float widthOfViewVolume, float heightOfViewVolume, float zNear, float zFar)
{
	_NBL_DEBUG_BREAK_IF(widthOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(heightOfViewVolume == 0.f); //division by zero
	_NBL_DEBUG_BREAK_IF(zNear == zFar); //division by zero

	matrix<T, 4, 4> m;
	m[0] = vector<T, 4>(2.f / widthOfViewVolume, 0.f, 0.f, 0.f);
	m[1] = vector<T, 4>(0.f, -2.f / heightOfViewVolume, 0.f, 0.f);
	m[2] = vector<T, 4>(0.f, 0.f, 1.f / (zFar - zNear), -zNear / (zFar - zNear));
	m[3] = vector<T, 4>(0.f, 0.f, 0.f, 1.f);

	return m;
}

}
}

#endif