#ifndef _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_TRAITS_INCLUDED_
#define _NBL_BUILTIN_HLSL_VECTOR_UTILS_VECTOR_TRAITS_INCLUDED_
#include <nbl/builtin/hlsl/cpp_compat/basic.h>

namespace nbl
{
namespace hlsl
{

// The whole purpose of this file is to enable the creation of partial specializations of the vector_traits for 
// custom types without introducing circular dependencies.

template<typename T>
struct vector_traits
{
    using scalar_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = 1u;
    NBL_CONSTEXPR_STATIC_INLINE bool IsVector = false;
};

// i choose to implement it this way because of this DXC bug: https://github.com/microsoft/DirectXShaderCom0piler/issues/7007
#define DEFINE_VECTOR_TRAITS_TEMPLATE_SPECIALIZATION(DIMENSION)\
template<typename T> \
struct vector_traits<vector<T, DIMENSION> >\
{\
    using scalar_type = T;\
    NBL_CONSTEXPR_STATIC_INLINE uint32_t Dimension = DIMENSION;\
    NBL_CONSTEXPR_STATIC_INLINE bool IsVector = true;\
};\

DEFINE_VECTOR_TRAITS_TEMPLATE_SPECIALIZATION(2)
DEFINE_VECTOR_TRAITS_TEMPLATE_SPECIALIZATION(3)
DEFINE_VECTOR_TRAITS_TEMPLATE_SPECIALIZATION(4)

}
}

#endif