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
    using vec_t = emulated_vector_t<T, RowCount>;
    using this_t = emulated_matrix<T, RowCount, ColumnCount>;
    using transposed_t = emulated_matrix<T, ColumnCount, RowCount>;

    NBL_CONSTEXPR_STATIC_INLINE uint32_t getRowCount() { return RowCount; }
    NBL_CONSTEXPR_STATIC_INLINE uint32_t getColumnCount() { return RowCount; }

    vec_t rows[RowCount];

    transposed_t getTransposed()
    {
        static nbl::hlsl::array_get<this_t::vec_t, T> getter;
        static nbl::hlsl::array_set<transposed_t::vec_t, T> setter;

        transposed_t output;
        for (int i = 0; i < RowCount; ++i)
        {
            for (int j = 0; j < ColumnCount; ++j)
                setter(output.rows[i], j, getter(rows[j], i));
        }

        return output;
    }

    //vec_t operator[](uint32_t rowIdx)
    //{
    //    return rows[rowIdx];
    //}
};

template<typename EmulatedType>
using emulated_matrix_t2x2 = emulated_matrix<EmulatedType, 2, 2>;
template<typename EmulatedType>
using emulated_matrix_t3x3 = emulated_matrix<EmulatedType, 3, 3>;
template<typename EmulatedType>
using emulated_matrix_t4x4 = emulated_matrix<EmulatedType, 4, 4>;
template<typename EmulatedType>
using emulated_matrix_t3x4 = emulated_matrix<EmulatedType, 3, 4>;
}
}
#endif