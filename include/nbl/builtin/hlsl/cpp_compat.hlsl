#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_INCLUDED_

#include <nbl/builtin/hlsl/macros.h>

#ifndef __HLSL_VERSION
#include <type_traits>
#include <bit>

#define ARROW ->
#define NBL_CONSTEXPR constexpr
#define NBL_CONSTEXPR_STATIC constexpr static
#define NBL_CONSTEXPR_STATIC_INLINE constexpr static inline
#define NBL_CONST_MEMBER_FUNC const

#define NBL_ALIAS_TEMPLATE_FUNCTION(origFunctionName, functionAlias) \
template<typename... Args> \
inline auto functionAlias(Args&&... args) -> decltype(origFunctionName(std::forward<Args>(args)...)) \
{ \
    return origFunctionName(std::forward<Args>(args)...); \
}

namespace nbl::hlsl
{

template<typename T>
using add_reference = std::add_lvalue_reference<T>;

template<typename T>
using add_pointer = std::add_pointer<T>;

}

// We need variadic macro in order to handle multi parameter templates because the 
// preprocessor parses the template parameters as different macro parameters.
#define NBL_REF_ARG(...) typename nbl::hlsl::add_reference<__VA_ARGS__ >::type
#define NBL_CONST_REF_ARG(...) typename nbl::hlsl::add_reference<std::add_const_t<__VA_ARGS__ >>::type

#else

#define ARROW .arrow().
#define NBL_CONSTEXPR const static
#define NBL_CONSTEXPR_STATIC_INLINE const static
#define NBL_CONST_MEMBER_FUNC 

namespace nbl
{
namespace hlsl
{

#if 0 // TODO: for later
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
#endif

}
}

#define NBL_REF_ARG(...) inout __VA_ARGS__
#define NBL_CONST_REF_ARG(...) const in __VA_ARGS__

#endif

// it includes vector and matrix
#include <nbl/builtin/hlsl/cpp_compat/intrinsics.h>

#endif