#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#include <nbl/builtin/hlsl/math/quaternion/quaternion.hlsl>
// TODO: remove this header when deleting vectorSIMDf.hlsl
#ifndef __HLSL_VERSION
#include <nbl/core/math/glslFunctions.h>
#include "vectorSIMD.h"
#endif
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include "nbl/builtin/hlsl/cpp_compat/unroll.hlsl"

namespace nbl
{
namespace hlsl
{

template<typename MatT>
MatT diagonal(float diagonal = 1)
{
	MatT output;

	NBL_UNROLL_LIMITED(4)
	for (uint32_t i = 0; i < matrix_traits<MatT>::RowCount; ++i)
		NBL_UNROLL_LIMITED(4)
		for (uint32_t j = 0; j < matrix_traits<MatT>::ColumnCount; ++j)
			output[i][j] = 0;

	NBL_UNROLL_LIMITED(4)
	for (uint32_t diag = 0; diag < matrix_traits<MatT>::RowCount; ++diag)
		output[diag][diag] = diagonal;

	return output;
}

template<typename MatT>
MatT identity()
{
	// TODO
	// static_assert(MatT::Square);
	return diagonal<MatT>(1);
}

// TODO: this is temporary function, delete when removing vectorSIMD
#ifndef __HLSL_VERSION
template<typename T>
inline core::vectorSIMDf transformVector(NBL_CONST_REF_ARG(matrix<T, 4, 4>) mat, NBL_CONST_REF_ARG(core::vectorSIMDf) vec)
{
	core::vectorSIMDf output;
	float32_t4 tmp;
	for (int i = 0; i < 4; ++i) // rather do that that reinterpret_cast for safety
		tmp[i] = output[i];

	for (int i = 0; i < 4; ++i)
		output[i] = hlsl::dot<float32_t4>(mat[i], tmp);

	return output;
}
#endif
template<typename T>
inline matrix<T, 4, 4> getMatrix3x4As4x4(NBL_CONST_REF_ARG(matrix<T, 3, 4>) mat)
{
	matrix<T, 4, 4> output;
	for (int i = 0; i < 3; ++i)
		output[i] = mat[i];
	output[3] = float32_t4(0.0f, 0.0f, 0.0f, 1.0f);

	return output;
}

template<typename T, int N>
inline matrix<T, 3, 3> getSub3x3(NBL_CONST_REF_ARG(matrix<T, N, 4>) mat)
{
	return matrix<T, 3, 3>(mat);
}

template<int N, int M>
inline matrix<float64_t, N, M> getAs64BitPrecisionMatrix(NBL_CONST_REF_ARG(matrix<float32_t, N, M>) mat)
{
	matrix<float64_t, N, M> output;
	for (int i = 0; i < N; ++i)
		output[i] = mat[i];

	return output;
}

namespace transformation_matrix_utils_impl
{
	// This function calculates determinant using the scalar triple product.
	template<typename T>
	inline T determinant_helper(NBL_CONST_REF_ARG(matrix<T, 3, 3>) mat, NBL_REF_ARG(vector<T, 3>) r1crossr2)
	{
		r1crossr2 = hlsl::cross(mat[1], mat[2]);
		return hlsl::dot(mat[0], r1crossr2);
	}
}

//! returs adjugate of the cofactor (sub 3x3) matrix
template<typename T, int N, int M>
inline matrix<T, 3, 3> getSub3x3TransposeCofactors(NBL_CONST_REF_ARG(matrix<T, N, M>) mat)
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

template<typename T, int N>
inline bool getSub3x3InverseTranspose(NBL_CONST_REF_ARG(matrix<T, N, 4>) matIn, NBL_CONST_REF_ARG(matrix<T, 3, 3>) matOut)
{
	matrix<T, 3, 3> matIn3x3 = getSub3x3(matIn);
	vector<T, 3> r1crossr2;
	T d = transformation_matrix_utils_impl::determinant_helper(matIn3x3, r1crossr2);
	if (abs(d) <= FLT_MIN)
		return false;
	auto rcp = T(1.0f)/d;

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
inline matrix<T, 3, 4> concatenateBFollowedByA(NBL_CONST_REF_ARG(matrix<T, 3, 4>) a, NBL_CONST_REF_ARG(const matrix<T, 3, 4>) b)
{
	// TODO
	// static_assert(N == 3 || N == 4);

	const matrix<T, 4, 4> a4x4 = getMatrix3x4As4x4<hlsl::float32_t>(a);
	const matrix<T, 4, 4> b4x4 = getMatrix3x4As4x4<hlsl::float32_t>(b);
	return matrix<T, 3, 4>(mul(a4x4, b4x4));
}

template<typename T, int N>
inline void setScale(NBL_REF_ARG(matrix<T, N, 4>) outMat, NBL_CONST_REF_ARG(vector<T, 3>) scale)
{
	// TODO
	// static_assert(N == 3 || N == 4);

	outMat[0][0] = scale[0];
	outMat[1][1] = scale[1];
	outMat[2][2] = scale[2];
}

//! Replaces curent rocation and scale by rotation represented by quaternion `quat`, leaves 4th row and 4th colum unchanged
template<typename T, int N>
inline void setRotation(NBL_REF_ARG(matrix<T, N, 4>) outMat, NBL_CONST_REF_ARG(nbl::hlsl::quaternion<T>) quat)
{
	// TODO
	//static_assert(N == 3 || N == 4);

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

template<typename T, int N>
inline void setTranslation(NBL_REF_ARG(matrix<T, N, 4>) outMat, NBL_CONST_REF_ARG(vector<T, 3>) translation)
{
	// TODO
	// static_assert(N == 3 || N == 4);

	outMat[0].w = translation.x;
	outMat[1].w = translation.y;
	outMat[2].w = translation.z;
}

}
}

#endif