// Copyright (C) 2018-2025 - DevSH Graphics Programming Sp. z O.O.
// This file is part of the "Nabla Engine".
// For conditions of distribution and use, see copyright notice in nabla.h

#ifndef _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_
#define _NBL_BUILTIN_HLSL_SAMPLING_QUANTIZED_SEQUENCE_INCLUDED_

#include "nbl/builtin/hlsl/cpp_compat.hlsl"
#include "nbl/builtin/hlsl/bit.hlsl"
#include "nbl/builtin/hlsl/glsl_compat/core.hlsl"
#include "nbl/builtin/hlsl/concepts/vector.hlsl"
#include "nbl/builtin/hlsl/vector_utils/vector_traits.hlsl"

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

template<typename Q, typename F>
struct encode_helper
{
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dim = Q::Dimension;
    using sequence_type = Q;
    using input_type = vector<F, Dim>;
    using uniform_storage_scalar_type = unsigned_integer_of_size_t<sizeof(F)>; 
    using uniform_storage_type = vector<uniform_storage_scalar_type, Dim>; // type that holds uint bit representation of a unorm that can have 1s in MSB (normalized w.r.t whole scalar)
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormMultiplier = (1u << (8u * size_of_v<uniform_storage_scalar_type> - 1u)) - 1u;

    static sequence_type __call(const input_type unormvec)
    {
        uniform_storage_type asuint;
        NBL_UNROLL for(uint16_t i = 0; i < Dim; i++)
            asuint[i] = uniform_storage_scalar_type(unormvec[i] * UNormMultiplier);
        return sequence_type::create(asuint);
    }
};

template<typename Q, typename F>
struct decode_before_scramble_helper
{
    using storage_scalar_type = typename Q::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dim = Q::Dimension;
    using uvec_type = vector<uint32_t, Dim>;
    using sequence_type = Q;
    using return_type = vector<F, Dim>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = unorm_constant<float32_t,8u*sizeof(storage_scalar_type)>::value;

    return_type operator()(const uvec_type scrambleKey)
    {
        uvec_type seqVal;
        NBL_UNROLL for(uint16_t i = 0; i < Dim; i++)
            seqVal[i] = val.get(i);
        seqVal ^= scrambleKey;
        return return_type(seqVal) * bit_cast<float_of_size_t<sizeof(storage_scalar_type)> >(UNormConstant);
    }

    sequence_type val;
};
template<typename Q, typename F>
struct decode_after_scramble_helper
{
    using storage_scalar_type = typename Q::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dim = Q::Dimension;
    using uvec_type = vector<uint32_t, Dim>;
    using sequence_type = Q;
    using return_type = vector<F, Dim>;
    NBL_CONSTEXPR_STATIC_INLINE uint32_t UNormConstant = unorm_constant<float32_t,sequence_type::BitsPerComponent>::value;

    return_type operator()(NBL_CONST_REF_ARG(sequence_type) scrambleKey)
    {
        sequence_type scramble;
        scramble.data = val.data ^ scrambleKey.data;

        uvec_type seqVal;
        NBL_UNROLL for(uint16_t i = 0; i < Dim; i++)
            seqVal[i] = scramble.get(i);
        return return_type(seqVal) * bit_cast<float_of_size_t<sizeof(storage_scalar_type)> >(UNormConstant);
    }

    sequence_type val;
};

template<typename T>
NBL_BOOL_CONCEPT SequenceSpecialization = concepts::UnsignedIntegral<typename vector_traits<T>::scalar_type> && size_of_v<typename vector_traits<T>::scalar_type> <= 4;
}

// all Dim=1
template<typename T> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T>)
struct QuantizedSequence<T, 1 NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T>) >
{
    using this_t = QuantizedSequence<T, 1>;
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = 8u*size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dimension = uint16_t(1u);

    static this_t create(const store_type value)
    {
        this_t seq;
        seq.data = value;
        return seq;
    }

    store_type get(const uint16_t idx) { assert(idx >= 0 && idx < 1); return data; }
    void set(const uint16_t idx, const store_type value) { assert(idx >= 0 && idx < 1); data = value; }

    template<typename F>
    static this_t encode(const vector<F, Dimension> value)
    {
        return impl::encode_helper<this_t,F>::__call(value);
    }

    template<typename F>
    vector<F,Dimension> decode(const vector<unsigned_integer_of_size_t<sizeof(F)>,Dimension> scrambleKey)
    {
        impl::decode_before_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }
    template<typename F>
    vector<F,Dimension> decode(NBL_CONST_REF_ARG(this_t) scrambleKey)
    {
        impl::decode_after_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }

    store_type data;
};

// uint16_t, uint32_t; Dim=2,3,4
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == 1 && Dim > 1 && Dim < 5) >
{
    using this_t = QuantizedSequence<T, Dim>;
    using store_type = T;
    using scalar_type = store_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = (uint16_t(8u) * size_of_v<scalar_type>) - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dimension = Dim;

    static this_t create(const vector<store_type, Dimension> value)
    {
        this_t seq;
        seq.data = store_type(0u);
        NBL_UNROLL for (uint16_t i = 0; i < Dimension; i++)
            seq.set(i, value[i]);
        return seq;
    }

    store_type get(const uint16_t idx)
    {
        assert(idx >= 0 && idx < Dim);
        return glsl::bitfieldExtract(data, BitsPerComponent * idx, BitsPerComponent);
    }

    void set(const uint16_t idx, const store_type value)
    {
        assert(idx >= 0 && idx < Dim);
        data = glsl::bitfieldInsert(data, value >> DiscardBits, BitsPerComponent * idx, BitsPerComponent);
    }

    template<typename F>
    static this_t encode(const vector<F, Dimension> value)
    {
        return impl::encode_helper<this_t,F>::__call(value);
    }

    template<typename F>
    vector<F,Dimension> decode(const vector<unsigned_integer_of_size_t<sizeof(F)>,Dimension> scrambleKey)
    {
        impl::decode_before_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }
    template<typename F>
    vector<F,Dimension> decode(NBL_CONST_REF_ARG(this_t) scrambleKey)
    {
        impl::decode_after_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }

    store_type data;
};

// Dim 2,3,4 matches vector dim
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(impl::SequenceSpecialization<T> && vector_traits<T>::Dimension == Dim && Dim > 1 && Dim < 5) >
{
    using this_t = QuantizedSequence<T, Dim>;
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = 8u*size_of_v<scalar_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dimension = Dim;

    static this_t create(const store_type value)
    {
        this_t seq;
        seq.data = value;
        return seq;
    }

    scalar_type get(const uint16_t idx) { assert(idx >= 0 && idx < Dim); return data[idx]; }
    void set(const uint16_t idx, const scalar_type value) { assert(idx >= 0 && idx < Dim); data[idx] = value; }

    template<typename F>
    static this_t encode(const vector<F, Dimension> value)
    {
        return impl::encode_helper<this_t,F>::__call(value);
    }

    template<typename F>
    vector<F,Dimension> decode(const vector<unsigned_integer_of_size_t<sizeof(F)>,Dimension> scrambleKey)
    {
        impl::decode_before_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }
    template<typename F>
    vector<F,Dimension> decode(NBL_CONST_REF_ARG(this_t) scrambleKey)
    {
        impl::decode_after_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }

    store_type data;
};

// uint32_t2; Dim=3 -- should never use uint16_t2 instead of uint32_t
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(is_same_v<T,uint32_t2> && Dim == 3)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(is_same_v<T,uint32_t2> && Dim == 3) >
{
    using this_t = QuantizedSequence<T, Dim>;
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = (uint16_t(8u) * size_of_v<scalar_type>) - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dimension = Dim;

    static this_t create(const vector<scalar_type, Dimension> value)
    {
        this_t seq;
        seq.data = hlsl::promote<store_type>(0u);
        NBL_UNROLL for (uint16_t i = 0; i < Dimension; i++)
            seq.set(i, value[i]);
        return seq;
    }

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
        const scalar_type trunc_val = value >> DiscardBits;
        if (idx == 0)   // x
            data[0] = glsl::bitfieldInsert(data[0], trunc_val, 0u, BitsPerComponent);
        else if (idx == 1)  // y
        {
            data[0] = glsl::bitfieldInsert(data[0], trunc_val, BitsPerComponent, DiscardBits);
            data[1] = glsl::bitfieldInsert(data[1], trunc_val >> DiscardBits, 0u, DiscardBits - 1u);
        }
        else    // z
            data[1] = glsl::bitfieldInsert(data[1], trunc_val, DiscardBits - 1u, BitsPerComponent);
    }

    template<typename F>
    static this_t encode(const vector<F, Dimension> value)
    {
        return impl::encode_helper<this_t,F>::__call(value);
    }

    template<typename F>
    vector<F,Dimension> decode(const vector<unsigned_integer_of_size_t<sizeof(F)>,Dimension> scrambleKey)
    {
        impl::decode_before_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }
    template<typename F>
    vector<F,Dimension> decode(NBL_CONST_REF_ARG(this_t) scrambleKey)
    {
        impl::decode_after_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }

    store_type data;
};

// uint16_t2; Dim=4 -- should use uint16_t4 instead of uint32_t2
template<typename T, uint16_t Dim> NBL_PARTIAL_REQ_TOP(is_same_v<T,uint16_t2> && Dim == 4)
struct QuantizedSequence<T, Dim NBL_PARTIAL_REQ_BOT(is_same_v<T,uint16_t2> && Dim == 4) >
{
    using this_t = QuantizedSequence<T, Dim>;
    using store_type = T;
    using scalar_type = typename vector_traits<T>::scalar_type;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t StoreBits = uint16_t(8u) * size_of_v<store_type>;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t BitsPerComponent = StoreBits / Dim;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t DiscardBits = (uint16_t(8u) * size_of_v<scalar_type>) - BitsPerComponent;
    NBL_CONSTEXPR_STATIC_INLINE uint16_t Dimension = Dim;

    static this_t create(const vector<scalar_type, Dimension> value)
    {
        this_t seq;
        seq.data = hlsl::promote<store_type>(0u);
        NBL_UNROLL for (uint16_t i = 0; i < Dimension; i++)
            seq.set(i, value[i]);
        return seq;
    }

    scalar_type get(const uint16_t idx)
    {
        assert(idx >= 0 && idx < 4);
        if (idx < 2) // x y
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
        const scalar_type trunc_val = value >> DiscardBits;
        if (idx < 2) // x y
        {
            data[0] = glsl::bitfieldInsert(data[0], trunc_val, BitsPerComponent * idx, BitsPerComponent);
        }
        else    // z w
        {
            data[1] = glsl::bitfieldInsert(data[1], trunc_val, BitsPerComponent * (idx & uint16_t(1u)), BitsPerComponent);
        }
    }

    template<typename F>
    static this_t encode(const vector<F, Dimension> value)
    {
        return impl::encode_helper<this_t,F>::__call(value);
    }

    template<typename F>
    vector<F,Dimension> decode(const vector<unsigned_integer_of_size_t<sizeof(F)>,Dimension> scrambleKey)
    {
        impl::decode_before_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }
    template<typename F>
    vector<F,Dimension> decode(NBL_CONST_REF_ARG(this_t) scrambleKey)
    {
        impl::decode_after_scramble_helper<this_t,F> helper;
        helper.val.data = data;
        return helper(scrambleKey);
    }

    store_type data;
};

// no uint16_t4, uint32_t4; Dim=2

}

}
}

#endif
