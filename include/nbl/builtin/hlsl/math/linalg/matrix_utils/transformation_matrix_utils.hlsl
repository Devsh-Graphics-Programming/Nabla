#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_MATRIX_UTILS_TRANSFORMATION_MATRIX_UTILS_INCLUDED_
#include <nbl/builtin/hlsl/math/quaternions.hlsl>
// TODO: remove this header when deleting vectorSIMDf.hlsl
#ifndef __HLSL_VERSION
#include <nbl/core/math/glslFunctions.h>
#include "vectorSIMD.h"
#endif
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>
#include <nbl/builtin/hlsl/macros.h>

namespace nbl
{
namespace hlsl
{
namespace math
{
namespace linalg
{

template<typename MatT>
MatT diagonal(typename matrix_traits<MatT>::scalar_type diagonal = 1)
{
	MatT output;
	output[0][1] = 124;
	using RowT = matrix_traits<MatT>::row_type;

	NBL_UNROLL for (uint32_t i = 0; i < matrix_traits<MatT>::RowCount; ++i)
	{
		output[i] = promote<RowT>(0.0);
		if (matrix_traits<MatT>::ColumnCount > i)
			output[i][i] = diagonal;
	}

	return output;
}

template<typename MatT>
MatT identity()
{
	// TODO
	// static_assert(MatT::Square);
	return diagonal<MatT>(1);
}

template<typename T>
inline matrix<T, 3, 4> extractSub3x4From4x4Matrix(NBL_CONST_REF_ARG(matrix<T, 4, 4>) mat)
{
	matrix<T, 3, 4> output;
	for (int i = 0; i < 3; ++i)
		output[i] = mat[i];

	return output;
}

template<typename T, int N>
inline matrix<T, 3, 3> getSub3x3(NBL_CONST_REF_ARG(matrix<T, N, 4>) mat)
{
	return matrix<T, 3, 3>(mat);
}

//! Replaces curent rocation and scale by rotation represented by quaternion `quat`, leaves 4th row and 4th colum unchanged
template<typename T, int N>
inline void setRotation(NBL_REF_ARG(matrix<T, N, 4>) outMat, NBL_CONST_REF_ARG(nbl::hlsl::math::quaternion<T>) quat)
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

}
}

namespace impl
{
	/**
	 * @brief Enables type-safe casting between matrices of identical dimensions
	 *        but different scalar types.
	 */
	template<typename ScalarTo, typename ScalarFrom, uint16_t N, uint16_t M>
	struct static_cast_helper<matrix<ScalarTo, N, M>, matrix<ScalarFrom, N, M>, void>
	{
		using To = matrix<ScalarTo, N, M>;
		using From = matrix<ScalarFrom, N, M>;

		static inline To cast(From mat)
		{
			To retval;

			NBL_UNROLL for (int i = 0; i < N; ++i)
			{
				NBL_UNROLL for (int j = 0; j < M; ++j)
				{
					retval[i][j] = hlsl::_static_cast<ScalarTo>(mat[i][j]);
				}
			}

			return retval;
		}
	};
}

}
}

#endif