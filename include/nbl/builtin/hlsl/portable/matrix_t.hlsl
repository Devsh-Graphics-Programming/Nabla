#ifndef _NBL_BUILTIN_HLSL_PORTABLE_MATRIX_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_PORTABLE_MATRIX_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/emulated/matrix_t.hlsl>
#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T, uint32_t N, uint32_t M, bool fundamental = is_fundamental<T>::value>
struct portable_matrix
{
    using type = matrix<T, N, M>;
};
#ifdef __HLSL_VERSION
template<typename T, uint32_t N, uint32_t M>
struct portable_matrix<T, N, M, false>
{
    using type = emulated_matrix<T, N, M>;
};
#endif

template<typename T, uint32_t N, uint32_t M>
using portable_matrix_t = typename portable_matrix<T, N, M>::type;

template<typename T>
using portable_matrix_t2x2 = portable_matrix_t<T, 2, 2>;
template<typename T>
using portable_matrix_t3x3 = portable_matrix_t<T, 3, 3>;


#ifdef __HLSL_VERSION
template<typename device_caps = void>
using portable_float64_t2x2 = portable_matrix_t2x2<portable_float64_t<device_caps> >;
template<typename device_caps = void>
using portable_float64_t3x3 = portable_matrix_t3x3<portable_float64_t<device_caps> >;
#else
template<typename device_caps = void>
using portable_float64_t2x2 = portable_matrix_t2x2<float64_t>;
template<typename device_caps = void>
using portable_float64_t3x3 = portable_matrix_t3x3<float64_t>;
#endif

namespace portable_matrix_impl
{
template<typename LhsT, typename RhsT>
struct mul_helper
{
    static inline RhsT multiply(LhsT lhs, RhsT rhs)
    {
        return mul(lhs, rhs);
    }
};

// TODO: portable instead of emulated? so no need for partial spec?
template<typename ComponentT, uint16_t RowCount, uint16_t ColumnCount>
struct mul_helper<emulated_matrix<ComponentT, RowCount, ColumnCount>, emulated_vector_t<ComponentT, RowCount> >
{
    using LhsT = emulated_matrix<ComponentT, RowCount, ColumnCount>;
    using RhsT = emulated_vector_t<ComponentT, RowCount>;

    static inline RhsT multiply(LhsT mat, RhsT vec)
    {
        nbl::hlsl::array_get<hlsl::emulated_vector_t4<hlsl::emulated_float64_t<true, true>>, hlsl::emulated_float64_t<true, true>> getter;
        nbl::hlsl::array_set<hlsl::emulated_vector_t4<hlsl::emulated_float64_t<true, true>>, hlsl::emulated_float64_t<true, true>> setter;

        emulated_vector_t<ComponentT, RowCount> output;
        for (int i = 0; i < RowCount; ++i)
            setter(output, i, nbl::hlsl::dot(mat.rows[i], vec));

        return output;
    }
};
}

// TODO: move to basic.hlsl?
// TODO: concepts, to ensure that LhsT is a matrix and RhsT is a vector type
template<typename LhsT, typename RhsT>
RhsT mul(LhsT lhs, RhsT rhs)
{
    return portable_matrix_impl::mul_helper<LhsT, RhsT>::multiply(lhs, rhs);
}

}
}

#endif