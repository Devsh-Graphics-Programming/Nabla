#ifndef _NBL_BUILTIN_HLSL_MATH_LINALG_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_LINALG_BASIC_INCLUDED_
// TODO: remove this header when deleting vectorSIMDf.hlsl
#ifndef __HLSL_VERSION
#include <nbl/core/math/glslFunctions.h>
#include "vectorSIMD.h"
#endif
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.hlsl>
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
inline matrix<T, NOut, MOut> truncate(NBL_CONST_REF_ARG(matrix<T, NIn, MIn>) inMatrix)
{
	matrix<T, NOut, MOut> retval;

	for (uint16_t i = 0; i < NOut; ++i)
		for (uint16_t j = 0; j < MOut; ++j)
			retval[i][j] = inMatrix[i][j];

	return retval;
}

namespace impl
{
template<uint16_t MOut, uint16_t MIn, typename T>
struct zero_expand_helper
{
	static vector<T, MOut> __call(const vector<T, MIn> inVec)
	{
		return vector<T, MOut>(inVec, vector<T, MOut - MIn>(0));
	}
};
template<uint16_t M, typename T>
struct zero_expand_helper<M,M,T>
{
	static vector<T, M> __call(const vector<T, M> inVec)
	{
		return inVec;
	}
};
}

template<uint16_t MOut, uint16_t MIn, typename T NBL_FUNC_REQUIRES(MOut >= MIn)
vector<T, MOut> zero_expand(vector<T, MIn> inVec)
{
	return impl::zero_expand_helper<MOut, MIn, T>::__call(inVec);
}

template <uint16_t NOut, uint16_t MOut, uint16_t NIn, uint16_t MIn, typename T NBL_FUNC_REQUIRES(NOut >= NIn && MOut >= MIn)
matrix<T, NOut, MOut> promote_affine(const matrix<T, NIn, MIn> inMatrix)
{
	matrix<T, NOut, MOut> retval;

	using out_row_t = hlsl::vector<T, MOut>;

	NBL_UNROLL for (uint32_t row_i = 0; row_i < NIn; row_i++)
	{
		retval[row_i] = zero_expand<MOut, MIn>(inMatrix[row_i]);
	}
	NBL_UNROLL for (uint32_t row_i = NIn; row_i < NOut; row_i++)
	{
		retval[row_i] = promote<out_row_t>(0.0);
		if (row_i < MOut)
			retval[row_i][row_i] = T(1.0);
	}
	return retval;
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

	template<typename ScalarTo, typename ScalarFrom, uint16_t N>
	struct static_cast_helper<vector<ScalarTo, N>, vector<ScalarFrom, N>, void>
	{
		using To = vector<ScalarTo, N>;
		using From = vector<ScalarFrom, N>;

		static inline To cast(From vec)
		{
			To retval;

			NBL_UNROLL for (int i = 0; i < N; ++i)
			{
				retval[i] = hlsl::_static_cast<ScalarTo>(vec[i]);
			}

			return retval;
		}
	};
}

}
}

#endif
