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

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask;

template<typename T, uint16_t Dim>
struct decode_mask<T, Dim, 0> : integral_constant<T, 1> {};

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask : integral_constant<T, (decode_mask<T, Dim, Bits - 1>::value << Dim) | T(1)> {};

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
NBL_CONSTEXPR T decode_mask_v = decode_mask<T, Dim, Bits>::value;

// Compile-time still a bit primitive in HLSL, we can support arbitrary-dimensional morton codes in C++ but HLSL's have to be hand coded
template <typename T, uint16_t Dim>
struct decode_masks_array;

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

#else
template<typename T>
struct decode_masks_array<T, 2>
{
    NBL_CONSTEXPR_STATIC_INLINE vector<T, 2> Masks = vector<T, 2>(decode_mask_v<T, 2>, decode_mask_v<T, 2> << T(1));
};
//template<typename T>
//NBL_CONSTEXPR_STATIC_INLINE vector<T, 2> decode_masks_array<T, 2>::Masks = vector<T, 2>(decode_mask_v<T, 2>, decode_mask_v<T, 2> << T(1));
#endif

} //namespace impl


template<typename I, uint16_t D NBL_PRIMARY_REQUIRES(concepts::IntegralScalar<I> && 1 < D && D < 5)
struct code
{
    using this_t = code<I, D>;
    using U = make_unsigned<I>;



    static this_t create(vector<I, D> cartesian)
    {
        //... TODO ...
        return this_t();
    }

    //operator+, operator-, operator>>, operator<<, and other bitwise ops

    U value;
};

} //namespace morton
} //namespace hlsl
} //namespace nbl



#endif