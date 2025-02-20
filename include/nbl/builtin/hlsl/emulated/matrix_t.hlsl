#ifndef _NBL_BUILTIN_HLSL_EMULATED_MATRIX_T_HLSL_INCLUDED_
#define _NBL_BUILTIN_HLSL_EMULATED_MATRIX_T_HLSL_INCLUDED_

#include <nbl/builtin/hlsl/portable/float64_t.hlsl>
#include <nbl/builtin/hlsl/matrix_utils/matrix_traits.hlsl>

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

    transposed_t getTransposed() NBL_CONST_MEMBER_FUNC
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

// i choose to implement it this way because of this DXC bug: https://github.com/microsoft/DirectXShaderCompiler/issues/7007
#define DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(ROW_COUNT, COLUMN_COUNT) \
template<typename T> \
struct matrix_traits<emulated_matrix<T, ROW_COUNT, COLUMN_COUNT> > \
{ \
    using scalar_type = T; \
    using row_type = emulated_vector_t<T, COLUMN_COUNT>; \
    using transposed_type = emulated_matrix<T, COLUMN_COUNT, ROW_COUNT>; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = ROW_COUNT; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = COLUMN_COUNT; \
    NBL_CONSTEXPR_STATIC_INLINE bool Square = RowCount == ColumnCount; \
    NBL_CONSTEXPR_STATIC_INLINE bool IsMatrix = true; \
};

DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2, 2)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 3)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4, 4)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 4)

#undef DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION

template<typename T, int N, int M, int O>
struct mul_output<emulated_matrix<T, N, M>, emulated_matrix<T, M, O> >
{
    using type = emulated_matrix<T, N, O>;
};

template<typename T, int N, int M>
struct mul_output<emulated_matrix<T, N, M>, emulated_vector_t<T, N> >
{
    using type = emulated_vector_t<T, N>;
};

namespace cpp_compat_intrinsics_impl
{
template<typename T, int N, int M>
struct transpose_helper<emulated_matrix<T, N, M> >
{
    using transposed_t = typename matrix_traits<emulated_matrix<T, N, M> >::transposed_type;

	static transposed_t __call(NBL_CONST_REF_ARG(emulated_matrix<T, N, M>) m)
	{
        return m.getTransposed();
	}
};

template<typename ComponentT, int N, int M, int O>
struct mul_helper<emulated_matrix<ComponentT, N, M>, emulated_matrix<ComponentT, M, O> >
{
    using LhsT = emulated_matrix<ComponentT, N, M>;
    using RhsT = emulated_matrix<ComponentT, M, O>;
    using OutputT = typename mul_output<LhsT, RhsT>::type;

    static inline OutputT __call(LhsT lhs, RhsT rhs)
    {
        typename matrix_traits<RhsT>::transposed_type rhsTransposed = rhs.getTransposed();
        const uint32_t outputRowCount = matrix_traits<OutputT>::RowCount;
        const uint32_t outputColumnCount = matrix_traits<OutputT>::ColumnCount;
        using OutputVecType = typename matrix_traits<OutputT>::row_type;

        nbl::hlsl::array_set<OutputVecType, typename vector_traits<OutputVecType>::scalar_type> setter;

        OutputT output;
        for (int r = 0; r < outputRowCount; ++r)
        {
            for (int c = 0; c < outputColumnCount; ++c)
                setter(output.rows[r], c, dot<OutputVecType>(lhs.rows[r], rhsTransposed.rows[c]));
        }

        return output;
    }
};

template<typename ComponentT, uint16_t RowCount, uint16_t ColumnCount>
struct mul_helper<emulated_matrix<ComponentT, RowCount, ColumnCount>, emulated_vector_t<ComponentT, ColumnCount> >
{
    using MatT = emulated_matrix<ComponentT, RowCount, ColumnCount>;
    using VecT = emulated_vector_t<ComponentT, ColumnCount>;
    using OutVecT = emulated_vector_t<ComponentT, RowCount>;

    static inline OutVecT __call(MatT mat, VecT vec)
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

}
}
#endif