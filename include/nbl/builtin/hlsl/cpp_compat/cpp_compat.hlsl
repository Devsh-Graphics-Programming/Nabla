#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_

#ifndef __HLSL_VERSION
#include <type_traits>

#define ARROW ->
#define NBL_CONSTEXPR constexpr
#define NBL_CONSTEXPR_STATIC_INLINE constexpr static inline

namespace nbl::hlsl
{

template<typename T>
using add_reference = std::add_lvalue_reference<T>;

template<typename T>
using add_pointer = std::add_pointer<T>;

}

#define NBL_REF_ARG(T) nbl::hlsl::add_reference<T>::type
#define NBL_CONST_REF_ARG(T) nbl::hlsl::add_reference<std::add_const_t<T>>::type

#else

#define ARROW .arrow().
#define NBL_CONSTEXPR const static
#define NBL_CONSTEXPR_STATIC_INLINE const static

namespace nbl
{
namespace hlsl
{

template<typename T>
struct add_reference
{
  using type = ref<T>;
};
template<typename T>
struct add_pointer
{
  using type = ptr<T>;
};

}
}

#define NBL_REF_ARG(T) inout T
#define NBL_CONST_REF_ARG(T) const in T

#endif

#endif