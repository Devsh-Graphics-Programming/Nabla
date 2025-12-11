// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_

#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"
#include "nbl/builtin/hlsl/random/pcg.hlsl"

namespace nbl
{
namespace hlsl
{
namespace sampling
{

template<typename T, uint16_t Dim NBL_STRUCT_CONSTRAINABLE>
struct QuantizedSequence;


namespace impl
{
template<uint16_t Bits>
struct unorm_constant;
template<>
struct unorm_constant<4> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3d888889u; };
template<>
struct unorm_constant<5> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3d042108u; };
template<>
struct unorm_constant<8> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3b808081u; };
template<>
struct unorm_constant<10> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3a802008u; };
template<>
struct unorm_constant<16> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x37800080u; };
template<>
struct unorm_constant<21> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x35000004u; };
template<>
struct unorm_constant<32> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x2f800004u; };

template<typename T, uint16_t D, bool EncodeScramble>
struct decode_helper;

template<typename T, uint16_t D>
struct decode_helper<T, D, false>
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using fp_type = typename float_of_size<sizeof(scalar_type)>::type;
    using uvec_type = vector<scalar_type, D>;
    using sequence_type = QuantizedSequence<T, D>;
    using return_type = vector<fp_type, D>;
    NBL_CONSTEXPR_STATIC_INLINE scalar_type UNormConstant = unorm_constant<8u*sizeof(scalar_type)>::value;

    static return_type __call(NBL_CONST_REF_ARG(sequence_type) val, const uint32_t scrambleSeed)
    {
        random::PCG32 pcg = random::PCG32::construct(scrambleSeed);
        uvec_type seqVal;
        NBL_UNROLL for(uint16_t i = 0; i < D; i++)
            seqVal[i] = val.get(i) ^ pcg();
        return return_type(seqVal) * bit_cast<fp_type>(UNormConstant);
    }
};
template<typename T, uint16_t D>
struct decode_helper<T, D, true>
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using fp_type = typename float_of_size<sizeof(scalar_type)>::type;
    using uvec_type = vector<scalar_type, D>;
    using sequence_type = QuantizedSequence<T, D>;
    using sequence_store_type = typename sequence_type::store_type;
    using sequence_scalar_type = typename vector_traits<sequence_store_type>::scalar_type;
    using return_type = vector<fp_type, D>;
    NBL_CONSTEXPR_STATIC_INLINE scalar_type UNormConstant = sequence_type::UNormConstant;

    static return_type __call(NBL_CONST_REF_ARG(sequence_type) val, const uint32_t scrambleSeed)
    {
        random::PCG32 pcg = random::PCG32::construct(scrambleSeed);

        sequence_store_type scrambleKey;
        NBL_UNROLL for(uint16_t i = 0; i < vector_traits<sequence_store_type>::Dimension; i++)
            scrambleKey[i] = sequence_scalar_type(pcg());

        sequence_type scramble;
        scramble.data = scrambleKey ^ val.data;

        // sequence_type scramble;
        // NBL_UNROLL for(uint16_t i = 0; i < D; i++)
        //     scramble.set(i, pcg());
        // scramble.data ^= val.data;

        uvec_type seqVal;
        NBL_UNROLL for(uint16_t i = 0; i < D; i++)
            seqVal[i] = scramble.get(i);
        return return_type(seqVal) * bit_cast<fp_type>(UNormConstant);
    }
};
}

template<typename T, uint16_t D, bool EncodeScramble=false>
vector<typename float_of_size<sizeof(typename vector_traits<T>::scalar_type)>::type, D> decode(NBL_CONST_REF_ARG(QuantizedSequence<T, D>) val, const uint32_t scrambleSeed)
{
    return impl::decode_helper<T,D,EncodeScramble>::__call(val, scrambleSeed);
}

#define SEQUENCE_SPECIALIZATION_CONCEPT concepts::UnsignedIntegral<typename vector_traits<T>::scalar_type> && size_of_v<typename vector_traits<T>::scalar_type> <= 4

// all Dim=1
template<typename T> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT)
struct QuantizedSequence<T, 1 NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT) >
{
    using store_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<8u*sizeof(store_type)>::value;

    store_type get(const uint16_t idx) { assert(idx > 0 && idx < 1); return data; }
    void set(const uint16_t idx, const store_type value) { assert(idx > 0 && idx < 1); data = value; }

    store_type data;
};

// uint16_t, uint32_t; Dim=2,3,4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5) >
{
    using store_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = StoreBits - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<BitsPerComponent>::value;

    store_type get(const uint16_t idx)
    {
        assert(idx > 0 && idx < Dim);
        return (data >> (BitsPerComponent * idx)) & Mask;
    }

    void set(const uint16_t idx, const store_type value)
    {
        assert(idx > 0 && idx < Dim);
        const uint16_t bits = (BitsPerComponent * idx);
        data &= ~(Mask << bits);
        data |= ((value >> DiscardBits) & Mask) << bits;
    }

    store_type data;
};

// Dim 2,3,4 matches vector dim
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<8u*sizeof(scalar_type)>::value;

    scalar_type get(const uint16_t idx) { assert(idx > 0 && idx < Dim); return data[idx]; }
    void set(const uint16_t idx, const scalar_type value) { assert(idx > 0 && idx < Dim); data[idx] = value; }

    store_type data;
};

// uint16_t2, uint32_t2; Dim=3
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 3) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = StoreBits - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<BitsPerComponent>::value;

    scalar_type get(const uint16_t idx)
    {
        assert(idx > 0 && idx < 3);
        if (idx < 2)
        {
            return data[idx] & Mask;
        }
        else
        {
            scalar_type z = data[0] >> BitsPerComponent;
            z |= (data[1] >> BitsPerComponent) << (StoreBits-BitsPerComponent);
            return z;
        }
    }

    void set(const uint16_t idx, const scalar_type value)
    {
        assert(idx > 0 && idx < 3);
        if (idx < 2)
        {
            data[idx] &= ~Mask;
            data[idx] |= (value >> DiscardBits) & Mask;
        }
        else
        {
            const uint16_t zbits = StoreBits-BitsPerComponent;
            const uint16_t zmask = uint16_t(1u) << zbits;
            const scalar_type trunc_val = value >> DiscardBits;
            data[0] &= Mask;
            data[1] &= Mask;
            data[0] |= (trunc_val & zmask) << BitsPerComponent;
            data[1] |= (trunc_val >> (zbits) & zmask) << BitsPerComponent;
        }
    }

    store_type data;
};

// uint16_t2, uint32_t2; Dim=4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 4)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 2 && Dim == 4) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << BitsPerComponent) - uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = StoreBits - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<BitsPerComponent>::value;

    scalar_type get(const uint16_t idx)
    {
        assert(idx > 0 && idx < 4);
        const uint16_t i = (idx & uint16_t(2u)) >> uint16_t(1u);
        return (data[i] >> (BitsPerComponent * (idx & uint16_t(1u)))) & Mask;
    }

    void set(const uint16_t idx, const scalar_type value)
    {
        assert(idx > 0 && idx < 4);
        const uint16_t i = (idx & uint16_t(2u)) >> uint16_t(1u);
        const uint16_t odd = idx & uint16_t(1u);
        data[i] &= hlsl::mix(~Mask, Mask, bool(odd));
        data[i] |= ((value >> DiscardBits) & Mask) << (BitsPerComponent * odd);
    }

    store_type data;
};

// uint16_t4, uint32_t4; Dim=2
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 2)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 2) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using base_type = vector<scalar_type, 2>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<8u*sizeof(scalar_type)>::value;

    base_type get(const uint16_t idx)
    {
        assert(idx > 0 && idx < 2);
        base_type a;
        a[0] = data[uint16_t(2u) * idx];
        a[1] = data[uint16_t(2u) * idx + 1];
        return a;
    }

    void set(const uint16_t idx, const base_type value)
    {
        assert(idx > 0 && idx < 2);
        base_type a;
        data[uint16_t(2u) * idx] = value[0];
        data[uint16_t(2u) * idx + 1] = value[1];
    }

    store_type data;
};

// uint16_t4, uint32_t4; Dim=3
// uint16_t4 --> returns uint16_t2 - 21 bits per component: 16 in x, 5 in y
// uint16_t4 --> returns uint32_t2 - 42 bits per component: 32 in x, 10 in y
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(SEQUENCE_SPECIALIZATION_CONCEPT && vector_traits<T>::Dimension == 4 && Dim == 3) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using base_type = vector<scalar_type, 2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t LeftoverBitsPerComponent = BitsPerComponent - uint16_t(8u) * size_of_v<scalar_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Mask = (uint16_t(1u) << LeftoverBitsPerComponent) - uint16_t(1u);
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = StoreBits - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = impl::unorm_constant<8u*sizeof(scalar_type)>::value;

    base_type get(const uint16_t idx)
    {
        assert(idx > 0 && idx < 3);
        base_type a;
        a[0] = data[idx];
        a[1] = (data[3] >> (LeftoverBitsPerComponent * idx)) & Mask;
        return a;
    }

    void set(const uint16_t idx, const base_type value)
    {
        assert(idx > 0 && idx < 3);
        data[idx] = value[0];
        data[3] &= ~Mask;
        data[3] |= ((value[1] >> DiscardBits) & Mask) << (LeftoverBitsPerComponent * idx);
    }

    store_type data;
};

#undef SEQUENCE_SPECIALIZATION_CONCEPT

}

}
}

#endif
