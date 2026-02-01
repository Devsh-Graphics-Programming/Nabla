#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_MATRIX_UTILS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_MATRIX_UTILS_INCLUDED_
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
	return diagonal<MatT>(1);
}

template <uint16_t NOut, uint16_t MOut, uint16_t NIn, uint16_t MIn, typename T NBL_FUNC_REQUIRES(NOut <= NIn && MOut <= MIn && NOut != 0 && MOut != 0)
inline matrix<T, NOut, MOut> truncate(const NBL_CONST_REF_ARG(matrix<T, NIn, MIn>) inMatrix)
{
	matrix<T, NOut, MOut> retval;

	for (uint16_t i = 0; i < NOut; ++i)
		for (uint16_t j = 0; j < MOut; ++j)
			retval[i][j] = inMatrix[i][j];

	return retval;
}

template<typename T, int N>
inline matrix<T, 3, 3> getSub3x3(NBL_CONST_REF_ARG(matrix<T, N, 4>) mat)
{
	return matrix<T, 3, 3>(mat);
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
