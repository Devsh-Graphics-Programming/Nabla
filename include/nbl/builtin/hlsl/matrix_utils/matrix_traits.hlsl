#ifndef _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATRIX_UTILS_MATRIX_TRAITS_INCLUDED_

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

namespace nbl
{
namespace hlsl
{

template<typename MatT>
struct matrix_traits;

// i choose to implement it this way because of this DXC bug: https://github.com/microsoft/DirectXShaderCompiler/issues/7007
#define DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(ROW_COUNT, COLUMN_COUNT) \
template<typename T> \
struct matrix_traits<matrix<T, ROW_COUNT, COLUMN_COUNT> > \
{ \
    using ComponentType = T; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = ROW_COUNT; \
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = COLUMN_COUNT; \
    NBL_CONSTEXPR_STATIC_INLINE bool Square = RowCount == ColumnCount; \
};

DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2, 2)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 3)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4, 4)
DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3, 4)

// TODO: when this bug: https://github.com/microsoft/DirectXShaderCompiler/issues/7007 is fixed, uncomment and delete template specializations
/*template<typename T, uint32_t N, uint32_t M>
struct matrix_traits<matrix<T,N,M> >
{
    using ComponentType = T;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t RowCount = ROW_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t ColumnCount = COLUMN_COUNT;
    NBL_CONSTEXPR_STATIC_INLINE bool Square = RowCount == ColumnCount;
};
*/

}
}

#endif