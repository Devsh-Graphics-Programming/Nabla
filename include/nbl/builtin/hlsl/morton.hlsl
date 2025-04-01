#ifndef _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_
#define _NBL_BUILTIN_HLSL_MATH_MORTON_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/concepts/core.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/functional.hlsl"
#include "nbl/builtin/hlsl/emulated/uint64_t.hlsl"

namespace nbl
{
namespace hlsl
{
namespace morton
{

namespace impl
{

// Valid dimension for a morton code
template <uint16_t D>
NBL_BOOL_CONCEPT MortonDimension = 1 < D && D < 5;

// Masks

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
struct decode_mask;

template<typename T, uint16_t Dim>
struct decode_mask<T, Dim, 1> : integral_constant<T, 1> {};

template<typename T, uint16_t Dim, uint16_t Bits>
struct decode_mask : integral_constant<T, (decode_mask<T, Dim, Bits - 1>::value << Dim) | T(1)> {};

template<typename T, uint16_t Dim, uint16_t Bits = 8 * sizeof(T) / Dim>
NBL_CONSTEXPR T decode_mask_v = decode_mask<T, Dim, Bits>::value;

// Decode masks are different for each dimension

template<typename T, uint16_t Dim, uint16_t Bits, bool Vectorized = false>
struct MortonDecoder;

} //namespace impl

// Up to D = 4 supported
#define NBL_HLSL_MORTON_MASKS(U, D) _static_cast<vector<U,D> > (vector< U , 4 >(impl::decode_mask_v< U , D >,\
                                                    impl::decode_mask_v< U , D > << U (1),\
                                                    impl::decode_mask_v< U , D > << U (2),\
                                                    impl::decode_mask_v< U , D > << U (3)\
                                                   ))

// Making this even slightly less ugly is blocked by https://github.com/microsoft/DirectXShaderCompiler/issues/7006
// In particular, `Masks` should be a `const static` member field instead of appearing in every method using it
template<bool Signed, uint16_t Bits, uint16_t D, typename _uint64_t = uint64_t NBL_PRIMARY_REQUIRES(impl::MortonDimension<D> && D * Bits <= 64)
struct code
{
    using this_t = code<Signed, Bits, D, _uint64_t>;
    NBL_CONSTEXPR_STATIC uint16_t TotalBitWidth = D * Bits;
    using storage_t = conditional_t<(TotalBitWidth>16), conditional_t<(TotalBitWidth>32), _uint64_t, uint32_t>, uint16_t> ;

    
    storage_t value;
};

// Don't forget to delete this macro after usage
#undef NBL_HLSL_MORTON_MASKS

} //namespace morton
} //namespace hlsl
} //namespace nbl

#endif