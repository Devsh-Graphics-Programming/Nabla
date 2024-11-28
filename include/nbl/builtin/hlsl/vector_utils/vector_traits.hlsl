#ifndef _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_TRAITS_INCLUDED_
#include <nbl/builtin/hlsl/cpp_compat/basic.h>

namespace nbl
{
namespace hlsl
{

// The whole purpose of this file is to enable the creation of partial specializations of the vector_traits for 
// custom types without introducing circular dependencies.

template<typename VecT>
struct vector_traits;

// i choose to implement it this way because of this DXC bug: https://github.com/microsoft/DirectXShaderCom0piler/issues/7007
//#define DEFINE_VECTOR_TRAITS_TEMPLATE_SPECIALIZATION(DIMENSION) \
//template<typename T> \
//struct vector_traits<vector<T, DIMENSION> > \
//{ \
//    using ComponentType = T; \
//    NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = DIMENSION; \
//};\

//template<typename T, int N>
//struct vector_traits<vector<T, N> >
//{
//    using ComponentType = T;
//    NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = N;
//};

//DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(2)
//DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(3)
//DEFINE_MATRIX_TRAITS_TEMPLATE_SPECIALIZATION(4)

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