// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<uint16_t BytesLog2, int32_t Dim NBL_STRUCT_CONSTRAINABLE>
struct QuantizedSequence;

// byteslog2 = 1,2; dim = 1
template<uint16_t BytesLog2> NBL_PARTIAL_REQ_TOP(BytesLog2 > 0 && BytesLog2 < 3)
struct QuantizedSequence<BytesLog2, 1 NBL_PARTIAL_REQ_BOT(BytesLog2 > 0 && BytesLog2 < 3) >
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_store_bytes = uint16_t(1u) << BytesLog2;
    using base_store_type = typename unsigned_integer_of_size<base_store_bytes>::type;

    base_store_type getX() { return data; }
    void setX(const base_store_type value) { data = value; }

    base_store_type data;
};

// byteslog2 = 3,4; dim = 1
template<uint16_t BytesLog2> NBL_PARTIAL_REQ_TOP(BytesLog2 > 2 && BytesLog2 < 5)
struct QuantizedSequence<BytesLog2, 1 NBL_PARTIAL_REQ_BOT(BytesLog2 > 2 && BytesLog2 < 5) >
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_bytes_log2 = uint16_t(2u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_store_bytes = uint16_t(1u) << base_bytes_log2;
    using base_store_type = typename unsigned_integer_of_size<base_store_bytes>::type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t num_components = uint16_t(1u) << (BytesLog2 - base_bytes_log2);
    using store_type = vector<base_store_type, num_components>;

    store_type getX() { return data; }
    void setX(const store_type value) { data = value; }

    store_type data;
};

// byteslog2 = 2,3; dim = 2
template<uint16_t BytesLog2> NBL_PARTIAL_REQ_TOP(BytesLog2 > 1 && BytesLog2 < 4)
struct QuantizedSequence<BytesLog2, 2 NBL_PARTIAL_REQ_BOT(BytesLog2 > 2 && BytesLog2 < 5) >
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_bytes_log2 = BytesLog2 - uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_store_bytes = uint16_t(1u) << base_bytes_log2;
    using base_store_type = typename unsigned_integer_of_size<base_store_bytes>::type;
    using store_type = vector<base_store_type, 2>;

    base_store_type getX() { return data[0]; }
    base_store_type getY() { return data[1]; }
    void setX(const base_store_type value) { data[0] = value; }
    void setY(const base_store_type value) { data[1] = value; }

    store_type data;
};

// byteslog2 = 1; dim = 2,3,4
template<uint16_t Dim> NBL_PARTIAL_REQ_TOP(Dim > 1 && Dim < 5)
struct QuantizedSequence<1, Dim NBL_PARTIAL_REQ_BOT(Dim > 1 && Dim < 5) >
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_store_bytes = uint16_t(1u) << uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t store_bits = uint16_t(8u) * base_store_bytes;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t bits_per_component = store_bits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t MASK = (uint16_t(1u) << bits_per_component) - uint16_t(1u);
    using base_store_type = uint16_t;

    base_store_type getX() { return data & MASK; }
    base_store_type getY() { return (data >> bits_per_component) & MASK; }
    template<typename C=bool_constant<2 < Dim> NBL_FUNC_REQUIRES(C::value && 2 < Dim)
    base_store_type getZ() { return (data >> (bits_per_component * uint16_t(2u))) & MASK; }
    template<typename C=bool_constant<3 < Dim> NBL_FUNC_REQUIRES(C::value && 3 < Dim)
    base_store_type getW() { return (data >> (bits_per_component * uint16_t(3u))) & MASK; }

    void setX(const base_store_type value)
    {
        data &= ~MASK;
        data |= value & MASK;
    }
    void setY(const base_store_type value)
    {
        const uint16_t mask = MASK << bits_per_component;
        data &= ~mask;
        data |= (value & MASK) << bits_per_component;
    }
    template<typename C=bool_constant<2 < Dim> NBL_FUNC_REQUIRES(C::value && 2 < Dim)
    void setZ(const base_store_type value)
    {
        const uint16_t bits = (bits_per_component * uint16_t(2u));
        const uint16_t mask = MASK << bits;
        data &= ~mask;
        data |= (value & MASK) << bits;
    }
    template<typename C=bool_constant<3 < Dim> NBL_FUNC_REQUIRES(C::value && 3 < Dim)
    void setW(const base_store_type value)
    {
        const uint16_t bits = (bits_per_component * uint16_t(3u));
        const uint16_t mask = MASK << bits;
        data &= ~mask;
        data |= (value & MASK) << bits;
    }

    base_store_type data;
};

// byteslog2 = 2,3; dim = 3
template<uint16_t BytesLog2> NBL_PARTIAL_REQ_TOP(BytesLog2 > 1 && BytesLog2 < 4)
struct QuantizedSequence<BytesLog2, 3 NBL_PARTIAL_REQ_BOT(BytesLog2 > 2 && BytesLog2 < 5) >
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_bytes_log2 = BytesLog2 - uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t base_store_bytes = uint16_t(1u) << base_bytes_log2;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t store_bits = uint16_t(8u) * base_store_bytes;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t bits_per_component = store_bits / uint16_t(3u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t MASK = (uint16_t(1u) << bits_per_component) - uint16_t(1u);
    using base_store_type = typename unsigned_integer_of_size<base_store_bytes>::type;
    using store_type = vector<base_store_type, 2>;

    base_store_type getX() { return data[0] & MASK; }
    base_store_type getY()
    {
        base_store_type y = data[0] >> bits_per_component;
        y |= (data[1] >> bits_per_component) << (store_bits-bits_per_component);
        return y;
    }
    base_store_type getZ() { return data[1] & MASK; }

    void setX(base_store_type x)
    {
        data[0] &= ~MASK;
        data[0] |= x & MASK;
    }
    void setY(base_store_type y)
    {
        const uint16_t ybits = store_bits-bits_per_component;
        const uint16_t ymask = uint16_t(1u) << ybits;
        data[0] &= MASK;
        data[1] &= MASK;
        data[0] |= (y & ymask) << bits_per_component;
        data[1] |= (y >> (ybits) & ymask) << bits_per_component;
    }
    void setZ(base_store_type z)
    {
        data[1] &= ~MASK;
        data[1] |= z & MASK;
    }

    store_type data;
};

// not complete because we're changing the template params next commit

}

}
}

#endif
