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

template<typename ComponentT, uint16_t RowCount, uint16_t ColumnCount>
struct mul_helper<emulated_matrix<ComponentT, RowCount, ColumnCount>, emulated_vector_t<ComponentT, RowCount> >
{
    using MatT = emulated_matrix<ComponentT, RowCount, ColumnCount>;
    using VecT = emulated_vector_t<ComponentT, RowCount>;

    static inline VecT multiply(MatT mat, VecT vec)
    {
        nbl::hlsl::array_get<VecT, scalar_of_t<VecT> > getter;
        nbl::hlsl::array_set<VecT, scalar_of_t<VecT> > setter;

        VecT output;
        for (int i = 0; i < RowCount; ++i)
            setter(output, i, nbl::hlsl::dot<VecT>(mat.rows[i], vec));

        return output;
    }
};
}

// TODO: concepts, to ensure that LhsT is a matrix and RhsT is a vector type
template<typename MatT, typename VecT>
VecT mul(MatT mat, VecT vec)
{
    return portable_matrix_impl::mul_helper<MatT, VecT>::multiply(mat, vec);
}

}
}

#endif