#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat/basic.h>
#include <nbl/builtin/hlsl/cpp_compat/matrix.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename T>
struct matrix_traits
{
    using scalar_type = T;
    using row_type = void;
    using transposed_type = void;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = 1;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = 1;
    NBL_CONSTEXPR_STATIC_INLINE bool Square = false;
    NBL_CONSTEXPR_STATIC_INLINE bool IsMatrix = false;
};

// i choose to implement it this way because of this DXC bug: https://github.com/microsoft/DirectXShaderCompiler/issues/7007
#define DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(ROW_COUNT, COLUMN_COUNT) \
template<typename T> \
struct matrix_traits<matrix<T, ROW_COUNT, COLUMN_COUNT> > \
{ \
    using scalar_type = T; \
    using row_type = vector<T, COLUMN_COUNT>; \
    using transposed_type = matrix<T, COLUMN_COUNT, ROW_COUNT>; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = ROW_COUNT; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = COLUMN_COUNT; \
    NBL_CONSTEXPR_STATIC_INLINE bool Square = RowCount == ColumnCount; \
    NBL_CONSTEXPR_STATIC_INLINE bool IsMatrix = true; \
};

DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(1, 2)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(1, 3)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(1, 4)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2, 1)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2, 2)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2, 3)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2, 4)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 1)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 2)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 3)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 4)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4, 1)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4, 2)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4, 3)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4, 4)

#undef DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION

// TODO: when this bug: https://github.com/microsoft/DirectXShaderCompiler/issues/7007 is fixed, uncomment and delete template specializations
/*template<typename T, uint32_t N, uint32_t M>
struct matrix_traits<matrix<T,N,M> >
{
    using scalar_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = ROW_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = COLUMN_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE bool Square = RowCount == ColumnCount;
};
*/

}
}

#endif