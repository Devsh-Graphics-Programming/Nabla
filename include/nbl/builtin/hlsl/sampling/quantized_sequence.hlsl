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
template<typename FloatScalar, uint16_t Bits>
struct unorm_constant;
template<>
struct unorm_constant<float,4> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3d888889u; };
template<>
struct unorm_constant<float,5> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3d042108u; };
template<>
struct unorm_constant<float,8> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3b808081u; };
template<>
struct unorm_constant<float,10> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x3a802008u; };
template<>
struct unorm_constant<float,16> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x37800080u; };
template<>
struct unorm_constant<float,21> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x35000004u; };
template<>
struct unorm_constant<float,32> { NBL_CONSTEXPR_STATIC_INLINE uint32_t value = 0x2f800004u; };

template<typename T, uint16_t D>
struct decode_before_scramble_helper
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using uvec_type = vector<uint32_t, D>;
    using sequence_type = QuantizedSequence<T, D>;
    using return_type = vector<float32_t, D>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = unorm_constant<float,8u*sizeof(scalar_type)>::value;

    static return_type __call(NBL_CONST_REF_ARG(sequence_type) val, const uvec_type scrambleKey)
    {
        uvec_type seqVal;
        NBL_UNROLL for(uint16_t i = 0; i < D; i++)
            seqVal[i] = val.get(i);
        seqVal ^= scrambleKey;
        return return_type(seqVal) * bit_cast<float_of_size_t<sizeof(scalar_type)> >(UNormConstant);
    }
};
template<typename T, uint16_t D>
struct decode_after_scramble_helper
{
    using scalar_type = typename vector_traits<T>::scalar_type;
    using uvec_type = vector<uint32_t, D>;
    using sequence_type = QuantizedSequence<T, D>;
    using return_type = vector<float32_t, D>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = unorm_constant<float,sequence_type::BitsPerComponent>::value;

    static return_type __call(NBL_CONST_REF_ARG(sequence_type) val, NBL_CONST_REF_ARG(sequence_type) scrambleKey)
    {
        sequence_type scramble;
        scramble.data = val.data ^ scrambleKey.data;

        uvec_type seqVal;
        NBL_UNROLL for(uint16_t i = 0; i < D; i++)
            seqVal[i] = scramble.get(i);
        return return_type(seqVal) * bit_cast<float_of_size_t<sizeof(scalar_type)> >(UNormConstant);
    }
};

template<typename T>
NBL_BOOL_CONCEPT SequenceSpecialization = concepts::UnsignedIntegral<typename vector_traits<T>::scalar_type> && size_of_v<typename vector_traits<T>::scalar_type> <= 4;
}

// post-decode scramble
template<typename R, typename T, uint16_t D>
vector<R,D> decode(NBL_CONST_REF_ARG(QuantizedSequence<T, D>) val, const vector<unsigned_integer_of_size_t<sizeof(R)>,D> scrambleKey)
{
    return impl::decode_before_scramble_helper<T,D>::__call(val, scrambleKey);
}

// pre-decode scramble
template<typename R, typename T, uint16_t D>
vector<R,D> decode(NBL_CONST_REF_ARG(QuantizedSequence<T, D>) val, NBL_CONST_REF_ARG(QuantizedSequence<T, D>) scrambleKey)
{
    return impl::decode_after_scramble_helper<T,D>::__call(val, scrambleKey);
}

// all Dim=1
template<typename T> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T>)
struct QuantizedSequence<T, 1 NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T>) >
{
    using store_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = 8u*size_of_v<store_type>;

    store_type get(const uint16_t idx) { assert(idx > 0 && idx < 1); return data; }
    void set(const uint16_t idx, const store_type value) { assert(idx > 0 && idx < 1); data = value; }

    store_type data;
};

// uint16_t, uint32_t; Dim=2,3,4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5) >
{
    using store_type = T;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;

    store_type get(const uint16_t idx)
    {
        assert(idx > 0 && idx < Dim);
        return glsl::bitfieldExtract(data, BitsPerComponent * idx, BitsPerComponent);
    }

    void set(const uint16_t idx, const store_type value)
    {
        assert(idx > 0 && idx < Dim);
        glsl::bitfieldInsert(data, value, BitsPerComponent * idx, BitsPerComponent);
    }

    store_type data;
};

// Dim 2,3,4 matches vector dim
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = 8u*size_of_v<scalar_type>;

    scalar_type get(const uint16_t idx) { assert(idx > 0 && idx < Dim); return data[idx]; }
    void set(const uint16_t idx, const scalar_type value) { assert(idx > 0 && idx < Dim); data[idx] = value; }

    store_type data;
};

// uint32_t2; Dim=3 -- should never use uint16_t2 instead of uint32_t
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && size_of_v<typename vector_traits<T>::scalar_type> == 4 && vector_traits<T>::Dimension == 2 && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && size_of_v<typename vector_traits<T>::scalar_type> == 4 && vector_traits<T>::Dimension == 2 && Dim == 3) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = (uint16_t(8u) * size_of_v<scalar_type>) - BitsPerComponent;

    scalar_type get(const uint16_t idx)
    {
        assert(idx >= 0 && idx < 3);
        if (idx == 0)   // x
            return glsl::bitfieldExtract(data[0], 0u, BitsPerComponent);
        else if (idx == 1)  // y
        {
            scalar_type y = glsl::bitfieldExtract(data[0], BitsPerComponent, DiscardBits);
            y |= glsl::bitfieldExtract(data[1], 0u, DiscardBits - 1u) << DiscardBits;
            return y;
        }
        else    // z
            return glsl::bitfieldExtract(data[1], DiscardBits - 1u, BitsPerComponent);
    }

    void set(const uint16_t idx, const scalar_type value)
    {
        assert(idx >= 0 && idx < 3);
        if (idx == 0)   // x
            glsl::bitfieldInsert(data[0], value, 0u, BitsPerComponent);
        else if (idx == 1)  // y
        {
            glsl::bitfieldInsert(data[0], value, BitsPerComponent, DiscardBits);
            glsl::bitfieldInsert(data[1], value >> DiscardBits, 0u, DiscardBits - 1u);
        }
        else    // z
            glsl::bitfieldInsert(data[1], value, DiscardBits - 1u, BitsPerComponent);
    }

    store_type data;
};

// uint16_t2; Dim=4 -- should use uint16_t4 instead of uint32_t2
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && size_of_v<typename vector_traits<T>::scalar_type> == 2 && vector_traits<T>::Dimension == 2 && Dim == 4)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && size_of_v<typename vector_traits<T>::scalar_type> == 2 && vector_traits<T>::Dimension == 2 && Dim == 4) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;

    scalar_type get(const uint16_t idx)
    {
        assert(idx >= 0 && idx < 4);
        if (idx >= 0 && idx < 2) // x y
        {
            return glsl::bitfieldExtract(data[0], BitsPerComponent * idx, BitsPerComponent);
        }
        else    // z w
        {
            return glsl::bitfieldExtract(data[1], BitsPerComponent * (idx & uint16_t(1u)), BitsPerComponent);
        }
    }

    void set(const uint16_t idx, const scalar_type value)
    {
        assert(idx >= 0 && idx < 4);
        if (idx >= 0 && idx < 2) // x y
        {
            glsl::bitfieldInsert(data[0], value, BitsPerComponent * idx, BitsPerComponent);
        }
        else    // z w
        {
            glsl::bitfieldInsert(data[1], value, BitsPerComponent * (idx & uint16_t(1u)), BitsPerComponent);
        }
    }

    store_type data;
};

// no uint16_t4, uint32_t4; Dim=2

// uint32_t4; Dim=3 --> returns uint32_t2 - 42 bits per component: 32 in x, 10 in y
// use uint32_t2 instead of uint16_t4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && size_of_v<typename vector_traits<T>::scalar_type> == 4 && vector_traits<T>::Dimension == 4 && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && size_of_v<typename vector_traits<T>::scalar_type> == 4 && vector_traits<T>::Dimension == 4 && Dim == 3) >
{
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    using base_type = vector<scalar_type, 2>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;

    base_type get(const uint16_t idx)
    {
        assert(idx >= 0 && idx < 3);
        if (idx == 0)   // x
        {
            base_type x;
            x[0] = data[0];
            x[1] = glsl::bitfieldExtract(data[1], 0u, 10u);
            return x;
        }
        else if (idx == 1)  // y
        {
            base_type y;
            y[0] = glsl::bitfieldExtract(data[1], 10u, 22u);
            y[0] |= glsl::bitfieldExtract(data[2], 0u, 10u) << 22u;
            y[1] = glsl::bitfieldExtract(data[2], 10u, 10u);
            return y;
        }
        else    // z
        {
            base_type z;
            z[0] = glsl::bitfieldInsert(data[2], 20u, 12u);
            z[0] |= glsl::bitfieldInsert(data[3], 0u, 20u) << 12u;
            z[1] = glsl::bitfieldInsert(data[3], 20u, 10u);
            return z;
        }
    }

    void set(const uint16_t idx, const base_type value)
    {
        assert(idx >= 0 && idx < 3);
        if (idx == 0)   // x
        {
            data[0] = value[0];
            glsl::bitfieldInsert(data[1], value[1], 0u, 10u);
        }
        else if (idx == 1)  // y
        {
            glsl::bitfieldInsert(data[1], value[0], 10u, 22u);
            glsl::bitfieldInsert(data[2], value[0] >> 22u, 0u, 10u);
            glsl::bitfieldInsert(data[2], value[1], 10u, 10u);
        }
        else    // z
        {
            glsl::bitfieldInsert(data[2], value[0], 20u, 12u);
            glsl::bitfieldInsert(data[3], value[0] >> 12u, 0u, 20u);
            glsl::bitfieldInsert(data[3], value[1], 20u, 10u);
        }
    }

    store_type data;
    // data[0] = | -- x 32 bits -- |
    // data[1] = MSB | -- y 22 bits -- | -- x 10 bits -- | LSB
    // data[2] = MSB | -- z 12 bits -- | -- y 20 bits -- | LSB
    // data[3] = | -- z 30 bits -- |
};

}

}
}

#endif
