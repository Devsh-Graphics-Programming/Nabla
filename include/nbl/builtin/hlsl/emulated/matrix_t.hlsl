#ifndef _NBL_BUILTIN_HLSL_EMULATED_MATRIX_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_MATRIX_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T, uint32_t RowCount, uint32_t ColumnCount>
struct emulated_matrix
{
    using vec_t = emulated_vector_t<T, ColumnCount>;
    using this_t = emulated_matrix<T, RowCount, ColumnCount>;
    using transposed_t = emulated_matrix<T, ColumnCount, RowCount>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t getRowCount() { return RowCount; }
    NBL_CONSTEXPR_STATIC_INLINE uint32_t getColumnCount() { return RowCount; }

    vec_t rows[RowCount];

    transposed_t getTransposed()
    {
        static nbl::hlsl::array_get<typename this_t::vec_t, T> getter;
        static nbl::hlsl::array_set<typename transposed_t::vec_t, T> setter;

        transposed_t output;
        for (int i = 0; i < RowCount; ++i)
        {
            for (int j = 0; j < ColumnCount; ++j)
                setter(output.rows[i], j, getter(rows[j], i));
        }

        return output;
    }

    inline vec_t operator[](uint32_t idx) { return rows[idx]; }
};

template<typename EmulatedType>
using emulated_matrix_t2x2 = emulated_matrix<EmulatedType, 2, 2>;
template<typename EmulatedType>
using emulated_matrix_t3x3 = emulated_matrix<EmulatedType, 3, 3>;
template<typename EmulatedType>
using emulated_matrix_t4x4 = emulated_matrix<EmulatedType, 4, 4>;
template<typename EmulatedType>
using emulated_matrix_t3x4 = emulated_matrix<EmulatedType, 3, 4>;

namespace emulated_matrix_impl
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
struct mul_helper<emulated_matrix<ComponentT, RowCount, ColumnCount>, emulated_vector_t<ComponentT, ColumnCount> >
{
    using MatT = emulated_matrix<ComponentT, RowCount, ColumnCount>;
    using VecT = emulated_vector_t<ComponentT, ColumnCount>;
    using OutVecT = emulated_vector_t<ComponentT, RowCount>;

    static inline OutVecT multiply(MatT mat, VecT vec)
    {
        nbl::hlsl::array_get<VecT, typename vector_traits<VecT>::scalar_type> getter;
        nbl::hlsl::array_set<VecT, typename vector_traits<VecT>::scalar_type> setter;

        OutVecT output;
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
    return emulated_matrix_impl::mul_helper<MatT, VecT>::multiply(mat, vec);
}

}
}
#endif