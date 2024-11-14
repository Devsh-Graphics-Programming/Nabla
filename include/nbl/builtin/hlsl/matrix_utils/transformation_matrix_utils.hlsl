#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>
#include <nbl/builtin/hlsl/math/quaternion/quaternion.hlsl>
// TODO: remove this header when deleting vectorSIMDf.hlsl
#include <nbl/core/math/glslFunctions.h>
#include "vectorSIMD.h"
#include <nbl/builtin/hlsl/concepts.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/concepts.hlsl>

namespace nbl
{
namespace hlsl
{
	// goal: identity<matrixType>::value
	// plan: diagonal<T, N, M>
	// identity<typename MatrixType>
	// 
	// partial spec:
	// template<T, N, M>
	// identity<matrix<T,N,M>

namespace transfromation_matrix_utils_impl
{
template<typename T, uint32_t N, uint32_t M>
matrix<T,N,M> diagonal(float diagonal)
{
	using MatT = matrix<T, N, M>;
	MatT output;

	for (int i = 0; i < matrix_traits<MatT>::RowCount; ++i)
		for (int j = 0; j < matrix_traits<MatT>::ColumnCount; ++j)
			output[i][j] = 0;

	auto a = matrix_traits<MatT>::RowCount;
	auto b = matrix_traits<MatT>::ColumnCount;

	for (int diag = 0; diag < matrix_traits<MatT>::RowCount; ++diag)
		output[diag][diag] = diagonal;

	return output;
};
}

template<typename MatT>
struct identity;

template<typename T, uint32_t N, uint32_t M>
struct identity<matrix<T, N, M> >
{
	static matrix<T, N, M> get()
	{
		return transfromation_matrix_utils_impl::diagonal<T, N, M>(1);
	}
};

#define IDENTITY_MATRIX(TYPE, N, M)\
const matrix<TYPE, N, M> TYPE ## N ## x ## M ## _identity = identity<matrix<TYPE, N, M> >::get();

#define DEFINE_IDENTITY_MATRICES(TYPE)\
IDENTITY_MATRIX(TYPE, 2, 2)\
IDENTITY_MATRIX(TYPE, 3, 3)\
IDENTITY_MATRIX(TYPE, 4, 4)\
IDENTITY_MATRIX(TYPE, 3, 4)

DEFINE_IDENTITY_MATRICES(float32_t)
DEFINE_IDENTITY_MATRICES(float64_t)
DEFINE_IDENTITY_MATRICES(int32_t)
DEFINE_IDENTITY_MATRICES(int64_t)
DEFINE_IDENTITY_MATRICES(uint32_t)
DEFINE_IDENTITY_MATRICES(uint64_t)

#undef DEFINE_IDENTITY_MATRICES
#undef IDENTITY_MATRIX

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

//! returs adjugate of the cofactor (sub 3x3) matrix
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

}
}

#endif