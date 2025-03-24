#ifndef _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/concepts/core.hlsl"

namespace nbl
{
namespace hlsl
{
namespace morton
{

namespace impl
{

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
struct decode_mask;

template<typename T, uint16_t Dim>
struct decode_mask<T, Dim, 0> : integral_constant<T, 1> {};

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask : integral_constant<T, (decode_mask<T, Dim, Bits - 1>::value << Dim) | T(1)> {};

#ifndef __HLSL_VERSION

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
NBL_CONSTEXPR T decode_mask_v = decode_mask<T, Dim, Bits>::value;

template <typename T, uint16_t Dim>
struct decode_masks_array
{
    static consteval vector<T, Dim> generateMasks()
    {
        vector<T, Dim> masks;
        for (auto i = 0u; i < Dim; i++)
        {
            masks[i] = decode_mask_v<T, Dim> << T(i);
        }
        return masks;
    }

    NBL_CONSTEXPR_STATIC_INLINE vector<T, Dim> Masks = generateMasks();
};

template <typename T, uint16_t Dim>
NBL_CONSTEXPR vector<T, Dim> decode_masks = decode_masks_array<T, Dim>::Masks;

#endif

} //namespace impl

// HLSL only supports up to D = 4, and even then having this in a more generic manner is blocked by a DXC issue targeting SPIR-V
#ifndef __HLSL_VERSION

#define NBL_HLSL_MORTON_MASKS(U, D) impl::decode_masks< U , D >

#else 

// Up to D = 4 supported
#define NBL_HLSL_MORTON_MASKS(U, D) vector< U , 4 >(impl::decode_mask< U , D >::value,\
                                                    impl::decode_mask< U , D >::value << U (1),\
                                                    impl::decode_mask< U , D >::value << U (2),\
                                                    impl::decode_mask< U , D >::value << U (3)\
                                                   )

#endif

// Making this even slightly less ugly is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
// In particular, `Masks` should be a `const static` member field instead of appearing in every method using it
template<typename I, uint16_t D NBL_PRIMARY_REQUIRES(concepts::IntegralScalar<I> && 1 < D && D < 5)
struct code
{
    using this_t = code<I, D>;
    using U = make_unsigned_t<I>;

#ifdef __HLSL_VERSION
    _Static_assert(is_same_v<U, uint32_t>,
        "make_signed<T> requires that T shall be a (possibly cv-qualified) "
        "integral type or enumeration but not a bool type.");
#endif

    static this_t create(vector<I, D> cartesian)
    {
        NBL_CONSTEXPR_STATIC_INLINE vector<I, D> Masks = NBL_HLSL_MORTON_MASKS(I, D);
        printf("%d %d %d %d", Masks[0], Masks[1], Masks[2], Masks[3]);
        this_t foo;
        foo.value = U(0);
        return foo;
    }

    //operator+, operator-, operator>>, operator<<, and other bitwise ops

    U value;
};

// Don't forget to delete this macro after usage
#undef NBL_HLSL_MORTON_MASKS

} //namespace morton
} //namespace hlsl
} //namespace nbl



#endif