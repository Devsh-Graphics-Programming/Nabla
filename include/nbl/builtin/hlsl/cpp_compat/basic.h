#ifndef _NBL_BUILTIN_HLSL_CPP_COMPAT_BASIC_INCLUDED_
#define _NBL_BUILTIN_HLSL_CPP_COMPAT_BASIC_INCLUDED_

#include <nbl/builtin/hlsl/macros.h>
#include <nbl/builtin/hlsl/concepts/impl/base.hlsl>

#ifndef __HLSL_VERSION
#include <type_traits>

#define ARROW ->
#define NBL_CONSTEXPR constexpr // TODO: rename to NBL_CONSTEXPR_VAR
#define NBL_CONSTEXPR_FUNC constexpr
#define NBL_CONSTEXPR_STATIC constexpr static
#define NBL_CONSTEXPR_INLINE constexpr inline
#define NBL_CONSTEXPR_STATIC_INLINE constexpr static inline
#define NBL_CONSTEXPR_STATIC_FUNC constexpr static
#define NBL_CONSTEXPR_INLINE_FUNC constexpr inline
#define NBL_CONSTEXPR_STATIC_INLINE_FUNC constexpr static inline
#define NBL_CONSTEXPR_FORCED_INLINE_FUNC NBL_FORCE_INLINE constexpr
#define NBL_CONST_MEMBER_FUNC const
#define NBL_IF_CONSTEXPR(...) if constexpr (__VA_ARGS__)

namespace nbl::hlsl
{
    template<typename T>
    using add_reference = std::add_lvalue_reference<T>;

    template<typename T>
    using add_pointer = std::add_pointer<T>;

}

// We need variadic macro in order to handle multi parameter templates because the 
// preprocessor parses the template parameters as different macro parameters.
#define NBL_REF_ARG(...) __VA_ARGS__&
#define NBL_CONST_REF_ARG(...) const __VA_ARGS__&

// NOTE: implementation below will mess up template parameter deduction
//#define NBL_REF_ARG(...) typename nbl::hlsl::add_reference<__VA_ARGS__ >::type
//#define NBL_CONST_REF_ARG(...) typename nbl::hlsl::add_reference<std::add_const_t<__VA_ARGS__ >>::type

#else


#define ARROW .arrow().
#define NBL_CONSTEXPR const static // TODO: rename to NBL_CONSTEXPR_VAR
#define NBL_CONSTEXPR_FUNC
#define NBL_CONSTEXPR_STATIC const static
#define NBL_CONSTEXPR_INLINE const static
#define NBL_CONSTEXPR_STATIC_INLINE const static
#define NBL_CONSTEXPR_STATIC_FUNC static
#define NBL_CONSTEXPR_INLINE_FUNC inline
#define NBL_CONSTEXPR_STATIC_INLINE_FUNC static inline
#define NBL_CONSTEXPR_FORCED_INLINE_FUNC inline
#define NBL_CONST_MEMBER_FUNC
#define NBL_IF_CONSTEXPR(...) if (__VA_ARGS__)

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

#define NBL_REF_ARG(...) [[vk::ext_reference]] inout __VA_ARGS__
#define NBL_CONST_REF_ARG(...) const in __VA_ARGS__

#endif

namespace nbl
{
namespace hlsl
{
namespace impl
{
template<typename To, typename From, typename Enabled = void NBL_STRUCT_CONSTRAINABLE >
struct static_cast_helper
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC To cast(NBL_CONST_REF_ARG(From) u)
    {
#ifndef __HLSL_VERSION
        return static_cast<To>(u);
#else
        return To(u);
#endif
    }
};

// CPP-side, this can invoke the copy constructor if the copy is non-trivial in generic code
// HLSL-side, this enables generic conversion code between types, contemplating the case where no conversion is needed
template<typename Same, typename Enabled>
struct static_cast_helper<Same, Same, Enabled>
{
    NBL_CONSTEXPR_STATIC_INLINE_FUNC Same cast(NBL_CONST_REF_ARG(Same) s)
    {
#ifndef __HLSL_VERSION
        return static_cast<Same>(s);
#else
        return s;
#endif
    }
};

}

template<typename To, typename From>
NBL_CONSTEXPR_INLINE_FUNC To _static_cast(NBL_CONST_REF_ARG(From) v)
{
    return impl::static_cast_helper<To, From>::cast(v);
}

}
}

#endif
