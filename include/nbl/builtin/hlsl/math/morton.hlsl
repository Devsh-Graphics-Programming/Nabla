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

// Valid dimension for a morton code
#ifndef __HLSL_VERSION

template <uint16_t D>
NBL_BOOL_CONCEPT MortonDimension = D > 1;

#else

template <uint16_t D>
NBL_BOOL_CONCEPT MortonDimension = 1 < D && D < 5;

#endif

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
struct decode_mask;

template<typename T, uint16_t Dim>
struct decode_mask<T, Dim, 0> : integral_constant<T, 1> {};

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask : integral_constant<T, (decode_mask<T, Dim, Bits - 1>::value << Dim) | T(1)> {};

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
NBL_CONSTEXPR T decode_mask_v = decode_mask<T, Dim, Bits>::value;

#ifndef __HLSL_VERSION

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
// This will throw a DXC warning about the vector being truncated - no way around that
#define NBL_HLSL_MORTON_MASKS(U, D) vector< U , 4 >(impl::decode_mask_v< U , D >,\
                                                    impl::decode_mask_v< U , D > << U (1),\
                                                    impl::decode_mask_v< U , D > << U (2),\
                                                    impl::decode_mask_v< U , D > << U (3)\
                                                   )

#endif

// Making this even slightly less ugly is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
// In particular, `Masks` should be a `const static` member field instead of appearing in every method using it
template<typename I, uint16_t D NBL_PRIMARY_REQUIRES(concepts::IntegralScalar<I> && impl::MortonDimension<D>)
struct code
{
    using this_t = code<I, D>;
    using U = make_unsigned_t<I>;

    static this_t create(vector<I, D> cartesian)
    {
        NBL_CONSTEXPR_STATIC_INLINE vector<U, D> Masks = NBL_HLSL_MORTON_MASKS(U, D);
        this_t foo;
        foo.value = Masks[0];
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