#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MUL_OUTPUT_T_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MUL_OUTPUT_T_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename LhsT, typename RhsT>
struct mul_output;

template<typename T, int N, int M>
struct mul_output<matrix<T, N, M>, vector<T, N> >
{
	using type = vector<T, N>;
};

template<typename T, int N, int M, int O>
struct mul_output<matrix<T, N, M>, matrix<T, M, O> >
{
	using type = matrix<T, N, O>;
};

//! type of matrix-matrix or matrix-vector multiplication
template<typename LhsT, typename RhsT>
using mul_output_t = typename mul_output<LhsT, RhsT>::type;

}
}

#endif